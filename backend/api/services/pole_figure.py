"""Backend pole-figure renderer (matplotlib-Agg, orix symmetry).

Projects symmetry-equivalent crystal poles into the sample frame, applies the
per-file R_user (right-multiply, non-destructive), and renders a multi-panel
scatter (+ optional density contour). Manual equal-area / stereographic
projection so we don't depend on orix's plot projection.
"""
from __future__ import annotations

import io
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

MAX_PANEL = 4  # guard against absurd hkl lists


def default_hkl_families(point_group) -> list[tuple[int, int, int]]:
    """Default pole families per crystal system (best-effort, cubic-first)."""
    name = getattr(point_group, "name", "") or ""
    # Cubic point groups: m-3m, 432, -43m, m-3, 23
    cubic = {"m-3m", "432", "-43m", "m-3", "23"}
    if name in cubic:
        return [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    # Hexagonal / trigonal → basal + prism + pyramidal (3-index hkl here).
    hexag = {"6/mmm", "6mm", "-6m2", "622", "6/m", "-6", "6",
             "-3m", "3m", "32", "-3", "3"}
    if name in hexag:
        return [(0, 0, 1), (1, 0, 0), (1, 0, 1)]
    # Fallback (tetragonal/orthorhombic/etc.)
    return [(0, 0, 1), (1, 0, 0), (1, 1, 0)]


def _project(xyz: np.ndarray, projection: str) -> np.ndarray:
    """Project unit vectors (N,3) on the +z hemisphere to 2D (N,2)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if projection == "stereographic":
        denom = 1.0 + z
        denom[denom < 1e-9] = 1e-9
        return np.column_stack([x / denom, y / denom])
    # equal_area (Lambert azimuthal), normalised so equator → radius 1.
    f = np.sqrt(1.0 / (1.0 + z + 1e-12))
    return np.column_stack([x * f, y * f])


def _density_grid(px: np.ndarray, py: np.ndarray, bins: int = 120, sigma: float = 2.5):
    """Smoothed point density on a grid over the unit disk → (Z, extent).

    2D histogram of the projected poles, gaussian-smoothed (scipy), masked
    outside the unit circle. Returned Z is normalised to its own max so the
    contour colour scale is comparable across panels.
    """
    from scipy.ndimage import gaussian_filter
    rng = [[-1.0, 1.0], [-1.0, 1.0]]
    h, xe, ye = np.histogram2d(px, py, bins=bins, range=rng)
    h = gaussian_filter(h, sigma=sigma)
    # mask outside the disk
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    h[(GX ** 2 + GY ** 2) > 1.0] = 0.0
    mx = h.max()
    if mx > 0:
        h = h / mx
    return h.T, [-1.0, 1.0, -1.0, 1.0]  # transpose for imshow/contourf x-y order


def _poles_in_sample_frame(orientations, hkl, phase, r_user):
    """Symmetry-equivalent poles of `hkl` in the sample frame, as (M,3) xyz."""
    from orix.vector import Miller
    m = Miller(hkl=[hkl], phase=phase).symmetrise(unique=True)
    # display orientations (right-multiply R_user; non-destructive)
    rot_disp = orientations.rotations if hasattr(orientations, "rotations") else orientations
    rot_disp = rot_disp * r_user
    poles = (~rot_disp).outer(m)              # Vector3d, shape (n_ori, n_sym)
    xyz = poles.data.reshape(-1, 3)
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    return xyz / norm


def compute_pole_figure(xmap, phase_id, hkl_families, r_user, plot_cfg,
                        mode: str = "both", subsample: int = 20000) -> bytes:
    if mode not in ("scatter", "density", "both"):
        raise ValueError("mode must be 'scatter', 'density' or 'both'")
    if len(hkl_families) == 0 or len(hkl_families) > MAX_PANEL:
        raise ValueError(f"hkl_families must have 1..{MAX_PANEL} entries")
    if phase_id not in set(np.unique(xmap.phase_id).tolist()):
        raise ValueError(f"phase_id {phase_id} not present in this result")
    phase = xmap.phases[phase_id]
    if phase.point_group is None:
        raise ValueError(f"phase {phase.name!r} has no point group — cannot build pole figure")

    mask = xmap.phase_id == phase_id
    sub = xmap[mask]
    n = sub.size
    if n == 0:
        raise ValueError("no indexed pixels for this phase")
    # Subsample orientations (not poles) for the scatter to bound cost.
    if n > subsample:
        idx = np.random.RandomState(0).choice(n, size=subsample, replace=False)
        rots = sub.rotations[idx]
    else:
        rots = sub.rotations

    hemi_sign = 1.0 if plot_cfg.get("hemisphere", "upper") == "upper" else -1.0
    projection = plot_cfg.get("projection", "equal_area")

    fig, axes = plt.subplots(1, len(hkl_families),
                             figsize=(3.2 * len(hkl_families), 3.6), dpi=110)
    if len(hkl_families) == 1:
        axes = [axes]
    try:
        for ax, hkl in zip(axes, hkl_families):
            xyz = _poles_in_sample_frame(rots, hkl, phase, r_user)
            z = xyz[:, 2] * hemi_sign
            keep = z >= 0
            xyz_h = xyz[keep].copy()
            xyz_h[:, 2] = np.abs(xyz_h[:, 2]) if hemi_sign > 0 else -np.abs(xyz_h[:, 2])
            if hemi_sign < 0:
                xyz_h[:, 2] = -xyz_h[:, 2]  # flip to +z for projection math
            pts = _project(xyz_h, projection)
            # Plot-convention axis handling.
            px, py = pts[:, 0], pts[:, 1]
            if plot_cfg.get("x_direction", "east") == "north":
                px, py = -py, px  # rotate axes 90° so X points up
            if plot_cfg.get("z_into_plane", False):
                px = -px
            if mode in ("density", "both") and px.size > 0:
                Z, extent = _density_grid(px, py)
                ax.imshow(Z, extent=extent, origin="lower", cmap="magma",
                          interpolation="bilinear", zorder=0)
            if mode in ("scatter", "both"):
                pt_alpha = 0.18 if mode == "both" else 0.35
                pt_color = "#ffffff" if mode == "both" else "#3b6fb0"
                ax.scatter(px, py, s=2, c=pt_color, alpha=pt_alpha, linewidths=0, zorder=1)
            circ = plt.Circle((0, 0), 1.0, fill=False, color="#444", lw=1.0)
            ax.add_patch(circ)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal"); ax.axis("off")
            ax.set_title("{" + "".join(str(i) for i in hkl) + "}", fontsize=11)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)
