"""
Phase Map Generator - Create publication-quality phase maps with configurable scalebars.

Accepts either:
- orix CrystalMap objects (from indexing results)
- Raw 2D numpy arrays (phase_id grids) with metadata

Provides configurable scalebar, legend, and export functions.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar

logger = logging.getLogger(__name__)


@dataclass
class ScaleBarConfig:
    """Configuration for scalebar rendering."""
    enabled: bool = True
    location: str = 'lower right'
    length_fraction: float = 0.25
    color: str = 'white'
    box_color: str = 'black'
    box_alpha: float = 0.6
    font_size: int = 10
    border_pad: float = 0.5
    sep: float = 5.0  # Separation between bar and label

    # Valid locations for UI dropdown
    VALID_LOCATIONS = [
        'upper left', 'upper center', 'upper right',
        'center left', 'center', 'center right',
        'lower left', 'lower center', 'lower right',
    ]


@dataclass
class PhaseInfo:
    """Metadata for a single phase."""
    phase_id: int
    name: str
    color: str  # Hex color string


@dataclass
class PhaseMapData:
    """All data needed to render a phase map."""
    phase_map_2d: np.ndarray  # Shape: (n_rows, n_cols), values are phase IDs
    phases: List[PhaseInfo]  # Phase metadata (id, name, color)
    step_x: float  # Pixel size in micrometers
    step_y: float  # Pixel size in micrometers
    title: str = "Phase Map"


# Default phase colors (Dracula-inspired palette)
DEFAULT_PHASE_COLORS = [
    '#ff5555',  # Red
    '#50fa7b',  # Green
    '#8be9fd',  # Cyan
    '#ffb86c',  # Orange
    '#bd93f9',  # Purple
    '#ff79c6',  # Pink
    '#f1fa8c',  # Yellow
    '#6272a4',  # Comment (grayish blue)
]


def compute_ipf_colors(xmap, direction_str: str = "Z", r_user=None,
                       rotations_override=None) -> np.ndarray:
    """Compute IPF (Inverse Pole Figure) colors from a CrystalMap.

    Each pixel is colored according to its crystallographic orientation
    projected into the inverse pole figure for the given sample direction.

    Parameters
    ----------
    xmap : orix.crystal_map.CrystalMap
        Crystal map with orientations and phase information.
    direction_str : str
        Reference direction: ``"Z"`` (surface normal, default),
        ``"X"`` or ``"Y"``.
    r_user : orix.quaternion.Rotation, optional
        Per-file sample-frame rotation applied (right-multiply,
        ``rotations * r_user``) at render time WITHOUT mutating the stored
        xmap. ``None`` or identity leaves the colouring byte-identical to the
        native frame. Same convention as
        ``orientation_frame.to_vendor_export_frame``.
    rotations_override : np.ndarray, optional
        (xmap.size, 4) quaternion array (w,x,y,z) used INSTEAD of the stored
        rotations, display-only (the xmap is never mutated). Used by the
        grain-stabilised IPF option: for low-symmetry Laue groups (e.g. m-3)
        the IPF colour key is discontinuous across its fundamental-sector
        boundary, so ~1° orientation noise flips pixel colours drastically
        (orange↔blue↔green speckle on perfectly smooth data). Colouring each
        pixel by its GRAIN-MEAN orientation removes that display artefact.

    Returns
    -------
    np.ndarray
        RGB image, shape ``(n_rows, n_cols, 3)``, float64 in ``[0, 1]``.
        Unindexed pixels (``phase_id == -1``) are gray.
    """
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d
    from orix.quaternion import Orientation, Rotation

    directions = {
        "Z": Vector3d.zvector(),
        "X": Vector3d.xvector(),
        "Y": Vector3d.yvector(),
    }
    direction = directions.get(direction_str, Vector3d.zvector())

    shape = xmap.shape
    if len(shape) == 2:
        n_rows, n_cols = shape
    elif len(shape) == 1:
        n_rows, n_cols = shape[0], 1
    else:
        n_rows, n_cols = xmap.size, 1
    n_total = n_rows * n_cols
    # Gray default for unindexed pixels (#44475a ≈ rgb(68,71,90)/255)
    rgb = np.full((n_total, 3), [0.267, 0.278, 0.353])

    missing_symmetry = []
    for phase_id in np.unique(xmap.phase_id):
        if phase_id == -1:
            continue
        phase = xmap.phases[phase_id]
        pg = phase.point_group

        # Fallback: derive point_group from space_group if available
        if pg is None and phase.space_group is not None:
            try:
                from orix.crystal_map import Phase as _Phase
                _tmp = _Phase(space_group=phase.space_group)
                pg = _tmp.point_group
                logger.info(
                    f"Phase {phase_id} ({phase.name!r}): derived point_group "
                    f"{pg} from space_group {phase.space_group.number}"
                )
            except Exception:
                pass

        if pg is None:
            # Collect the missing-symmetry case and continue colouring the
            # rest — we still need to know about ALL bad phases before we
            # raise. If only some phases are missing symmetry the caller
            # may want to render those as grey while the rest carry colour
            # (use the legacy /render path), but the new /layer endpoint
            # must fail loudly so the user knows the IPF is incomplete.
            missing_symmetry.append((int(phase_id), phase.name or f"phase_{phase_id}"))
            continue

        mask = xmap.phase_id == phase_id
        n_phase = int(mask.sum())
        logger.info(
            f"IPF coloring phase {phase_id} ({phase.name!r}): "
            f"{n_phase} pixels, point_group={pg}"
        )
        color_key = IPFColorKeyTSL(
            symmetry=pg, direction=direction
        )
        # Select this phase's pixels by phase_id (unique), NOT by name.
        # Spherical-indexing results can carry several phases with the
        # SAME name (chemically-similar candidate phases the indexer kept
        # apart). ``xmap[phase.name]`` would match ALL of them at once and
        # orix raises "command that only permits one phase". The boolean
        # phase_id mask isolates exactly one phase.
        subset = xmap[mask]
        base_rot = (Rotation(np.asarray(rotations_override)[mask])
                    if rotations_override is not None else subset.rotations)
        if r_user is not None and not np.allclose(
            np.asarray(r_user.data), np.asarray(Rotation.identity().data)
        ):
            # Apply the sample-frame rotation (right-multiply — same convention
            # as orientation_frame.to_vendor_export_frame) WITHOUT mutating the
            # stored xmap. Re-wrap as Orientation with the phase symmetry so the
            # colour key reduces into the fundamental zone correctly.
            rot_disp = base_rot * r_user
            oris = Orientation(rot_disp, symmetry=pg)
        else:
            oris = Orientation(base_rot, symmetry=pg)
        ipf_colors = color_key.orientation2color(oris)
        rgb[mask] = ipf_colors.reshape(-1, 3)

    if missing_symmetry:
        details = ", ".join(f"id={pid} name={name!r}" for pid, name in missing_symmetry)
        raise ValueError(
            "IPF coloring would render grey for phases with missing symmetry: "
            f"{details}. Cause is usually the indexing pipeline not propagating "
            "point_group from the master pattern / CIF. Fix the source rather "
            "than rendering misleading flat-grey IPF maps."
        )

    return rgb.reshape((n_rows, n_cols, 3))


def compute_ipf_colors_grain_consistent(
    xmap,
    direction_str: str = "Z",
    r_user=None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    mask=None,
    fallback_deg: float = 10.0,
) -> np.ndarray:
    """Grain-consistent IPF colouring (v2) — de-jitters low-symmetry IPF maps
    WITHOUT flattening intra-grain gradients.

    Problem: for low-symmetry Laue groups (e.g. ``m-3`` cubic approximants) the
    IPF colour key is DISCONTINUOUS across its fundamental-sector boundary, so
    symmetry-equivalent directions on either side get maximally different
    colours and ~1° orientation noise renders smooth data as colour speckle.
    v1 (``rotations_override`` = grain mean) killed the speckle but also hid
    every real gradient. v2 keeps EACH pixel's own direction and only routes a
    whole grain through the SAME side of the key discontinuity:

    1. Segment grains per phase modulo the supergroup holohedry (5°) — variant
       splits collapse, so a physical grain is one label.
    2. Per grain: branch-consistent mean orientation → grain-mean DIRECTION
       (same ``O * direction`` convention as :func:`compute_ipf_colors`),
       reduced into the fundamental sector by the standard orix call.
    3. Per pixel: among its symmetry-equivalent directions pick the one closest
       (max dot) to the reduced grain-mean direction, and colour it with the
       key's in-sector TSL colour math applied WITHOUT re-reduction (the maths
       extrapolates continuously for points slightly outside the sector). RGB
       is clipped to ``[0, 1]``.
    4. Fallback: if any pixel's chosen representative sits > ``fallback_deg``
       (default 10°) from the reduced grain mean — genuinely bent grain,
       outside the validated extrapolation range — that grain falls back to the
       standard per-pixel reduction (i.e. the :func:`compute_ipf_colors` path).
    5. Grains with < 2 px, phase pixels not assigned to any grain, and
       unindexed pixels all keep the standard-path / grey baseline.

    Display-only: the stored xmap is never mutated. Returns an RGB image of
    shape ``(n_rows, n_cols, 3)``, float64 in ``[0, 1]``.

    Parameters mirror :func:`compute_ipf_colors`; ``n_rows``/``n_cols``/``mask``
    describe the full display grid (ROI results carry a boolean ``mask`` whose
    ``sum()`` equals ``xmap.size``).
    """
    from orix.vector import Vector3d
    from orix.quaternion import Rotation
    from orix.plot.direction_color_keys._util import (
        polar_coordinates_in_sector,
        rgb_from_polar_coordinates,
    )
    from backend.spherical_gpu.pipeline.variant_unification import (
        segment_supergroup_grains,
    )
    from backend.spherical_gpu.pseudosym import _qmul, _sym_quats

    directions = {
        "Z": Vector3d.zvector(),
        "X": Vector3d.xvector(),
        "Y": Vector3d.yvector(),
    }
    direction = directions.get(direction_str, Vector3d.zvector())

    if n_rows is None or n_cols is None:
        shape = xmap.shape
        if len(shape) == 2:
            n_rows, n_cols = shape
        elif len(shape) == 1:
            n_rows, n_cols = shape[0], 1
        else:
            n_rows, n_cols = xmap.size, 1
    n = int(n_rows) * int(n_cols)

    # Baseline = STANDARD per-pixel colouring. It handles r_user, the grey
    # default for unindexed pixels, and the missing-symmetry ValueError exactly
    # as before. v2 only OVERRIDES the pixels of grains where it applies.
    baseline_flat = np.asarray(
        compute_ipf_colors(xmap, direction_str, r_user=r_user)
    ).reshape(-1, 3)

    qdata = np.asarray(xmap.rotations.data).reshape(-1, 4).astype(np.float64)
    pid = np.asarray(xmap.phase_id).reshape(-1)
    n_src = qdata.shape[0]

    # Grey default for unindexed pixels (matches compute_ipf_colors).
    rgb_full = np.full((n, 3), [0.267, 0.278, 0.353], dtype=np.float64)

    # Map xmap rows -> full-grid flat indices (same layout logic as the v1
    # grain-stabiliser). Two known layouts: full grid, or ROI (mask.sum()).
    if n_src == n:
        flat_of_row = np.arange(n)
    elif mask is not None and int(np.asarray(mask).sum()) == n_src:
        flat_of_row = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    else:
        # Unknown layout — cannot place on a 2D grid for grain segmentation.
        # Fail soft: return the standard baseline best-effort (display helper).
        m = min(n_src, n)
        rgb_full[:m] = baseline_flat[:m]
        return rgb_full.reshape(int(n_rows), int(n_cols), 3)

    rgb_full[flat_of_row] = baseline_flat

    full_q = np.full((n, 4), np.nan)
    full_q[flat_of_row] = qdata
    phase_full = np.full(n, -1, dtype=np.int64)
    phase_full[flat_of_row] = pid

    r_user_active = (
        r_user is not None
        and not np.allclose(
            np.asarray(r_user.data), np.asarray(Rotation.identity().data)
        )
    )

    for phase_id in np.unique(pid):
        if phase_id == -1:
            continue
        try:
            pg = xmap.phases[int(phase_id)].point_group
            pg_name = pg.name
            laue = pg.laue
            sector = laue.fundamental_sector
        except Exception:
            continue  # baseline already coloured / errored for this phase
        try:
            S_sym = _sym_quats(pg_name)
            labels = segment_supergroup_grains(
                full_q, phase_full, int(n_rows), int(n_cols),
                int(phase_id), pg_name, 5.0,
            )
        except Exception:
            continue  # display helper: fail soft, keep baseline

        for g in np.unique(labels[labels >= 0]):
            pix = np.flatnonzero(labels == g)
            if pix.size < 2:
                continue  # standard baseline stays

            # Branch-consistent grain-mean orientation (each pixel snapped to
            # the symmetry equivalent nearest the grain seed before averaging).
            ref = full_q[pix[0]]
            qs = np.empty((pix.size, 4))
            for k, i in enumerate(pix):
                cands = _qmul(S_sym, full_q[i][None, :])
                best = cands[int(np.argmax(np.abs(cands @ ref)))]
                qs[k] = -best if float(np.dot(best, ref)) < 0 else best
            mean_q = qs.mean(axis=0)
            mean_q /= max(float(np.linalg.norm(mean_q)), 1e-12)

            # Grain-mean DIRECTION (display convention O * direction, incl.
            # r_user), reduced into the fundamental sector by the standard call.
            mean_rot = Rotation(mean_q[None, :])
            if r_user_active:
                mean_rot = mean_rot * r_user
            mean_dir_red = (mean_rot * direction).in_fundamental_sector(laue)
            mref = np.asarray(mean_dir_red.unit.data).reshape(3)

            # Per-pixel directions and their symmetry equivalents.
            rot_pix = Rotation(full_q[pix])
            if r_user_active:
                rot_pix = rot_pix * r_user
            dir_pix = rot_pix * direction
            eq = laue.outer(dir_pix)                      # Vector3d (n_sym, m)
            eq_data = np.asarray(eq.unit.data)            # (n_sym, m, 3)
            dots = (eq_data * mref[None, None, :]).sum(-1)  # (n_sym, m)
            best_k = np.argmax(dots, axis=0)              # (m,)
            chosen = np.take_along_axis(
                eq_data, best_k[None, :, None], axis=0)[0]   # (m, 3)
            chosen_dot = np.take_along_axis(dots, best_k[None, :], axis=0)[0]
            worst_angle = float(
                np.degrees(np.arccos(np.clip(chosen_dot.min(), -1.0, 1.0)))
            )
            if worst_angle > fallback_deg:
                # Genuinely bent grain — outside validated extrapolation range.
                # Keep the honest standard per-pixel baseline for this grain.
                continue

            az, pol = polar_coordinates_in_sector(sector, Vector3d(chosen).unit)
            pol = 0.5 + pol / 2
            rgb_grain = np.clip(
                np.asarray(rgb_from_polar_coordinates(az, pol)).reshape(-1, 3),
                0.0, 1.0,
            )
            rgb_full[pix] = rgb_grain

    return rgb_full.reshape(int(n_rows), int(n_cols), 3)


def extract_phase_map_from_xmap(xmap) -> PhaseMapData:
    """Extract a PhaseMapData from an orix CrystalMap object.

    Parameters
    ----------
    xmap : orix.crystal_map.CrystalMap
        Crystal map with phase_id and shape attributes.

    Returns
    -------
    PhaseMapData
        Ready-to-render phase map data.
    """
    phase_map_1d = xmap.phase_id
    shape = xmap.shape
    if len(shape) == 2:
        n_rows, n_cols = shape
    elif len(shape) == 1:
        n_rows, n_cols = shape[0], 1
    else:
        n_rows, n_cols = xmap.size, 1

    phase_map_2d = phase_map_1d.reshape((n_rows, n_cols))

    # Extract phase info from orix PhaseList
    phases = []
    for pid in sorted(set(phase_map_1d)):
        if pid == -1:
            phases.append(PhaseInfo(phase_id=-1, name="Not indexed", color='#44475a'))
            continue
        phase = xmap.phases[pid]
        color = phase.color if hasattr(phase, 'color') else DEFAULT_PHASE_COLORS[pid % len(DEFAULT_PHASE_COLORS)]
        # Convert matplotlib color to hex if needed
        if not isinstance(color, str):
            color = matplotlib.colors.to_hex(color)
        phases.append(PhaseInfo(phase_id=pid, name=phase.name, color=color))

    # Get step size from scan_unit / dy / dx if available
    step_x = getattr(xmap, 'dx', 1.0)
    step_y = getattr(xmap, 'dy', 1.0)
    if hasattr(step_x, 'magnitude'):
        step_x = float(step_x.magnitude)
    if hasattr(step_y, 'magnitude'):
        step_y = float(step_y.magnitude)

    return PhaseMapData(
        phase_map_2d=phase_map_2d,
        phases=phases,
        step_x=float(step_x),
        step_y=float(step_y),
    )


def create_phase_map_from_array(
    phase_array: np.ndarray,
    phase_names: Optional[Dict[int, str]] = None,
    phase_colors: Optional[Dict[int, str]] = None,
    step_x: float = 1.0,
    step_y: float = 1.0,
) -> PhaseMapData:
    """Create PhaseMapData from a raw 2D numpy array.

    Parameters
    ----------
    phase_array : np.ndarray
        2D array of phase IDs, shape (n_rows, n_cols).
    phase_names : dict, optional
        Mapping of phase_id -> name. Defaults to "Phase {id}".
    phase_colors : dict, optional
        Mapping of phase_id -> hex color. Defaults to palette.
    step_x, step_y : float
        Pixel size in micrometers.

    Returns
    -------
    PhaseMapData
    """
    unique_ids = sorted(np.unique(phase_array))
    phases = []
    for i, pid in enumerate(unique_ids):
        name = (phase_names or {}).get(pid, f"Phase {pid}")
        color = (phase_colors or {}).get(pid, DEFAULT_PHASE_COLORS[i % len(DEFAULT_PHASE_COLORS)])
        phases.append(PhaseInfo(phase_id=pid, name=name, color=color))

    return PhaseMapData(
        phase_map_2d=phase_array,
        phases=phases,
        step_x=step_x,
        step_y=step_y,
    )


def render_phase_map(
    data: PhaseMapData,
    scalebar_config: Optional[ScaleBarConfig] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    show_legend: bool = True,
) -> Figure:
    """Render a phase map as a matplotlib Figure.

    Parameters
    ----------
    data : PhaseMapData
        Phase map data with spatial calibration.
    scalebar_config : ScaleBarConfig, optional
        Scalebar settings. Defaults to ScaleBarConfig().
    figsize : tuple, optional
        Figure size (width, height) in inches. Auto-calculated if None.
    dpi : int
        Figure resolution.
    show_legend : bool
        Whether to show phase legend.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if scalebar_config is None:
        scalebar_config = ScaleBarConfig()

    is_rgb = (data.phase_map_2d.ndim == 3 and data.phase_map_2d.shape[2] == 3)
    n_rows, n_cols = data.phase_map_2d.shape[:2]

    # Auto-calculate figure size to maintain aspect ratio
    if figsize is None:
        aspect = n_rows / max(n_cols, 1)
        width = 8.0
        height = width * aspect
        # Add space for legend (only for discrete phase maps)
        if show_legend and not is_rgb:
            height += 0.6
        figsize = (width, max(height, 3.0))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if is_rgb:
        # IPF mode: direct RGB display (no colormap needed)
        ax.imshow(
            data.phase_map_2d,
            interpolation='nearest',
            origin='upper',
            aspect='equal',
        )
    else:
        # Discrete phase map mode
        phase_ids = [p.phase_id for p in data.phases]
        colors = [p.color for p in data.phases]

        # Fail loudly on an empty phase list — the previous IndexError path
        # was replaced with a greyscale fallback that silently hid real
        # upstream bugs (all-unindexed xmap, extraction failures). Raise a
        # typed error the caller can map to a 422 with a meaningful message.
        if not phase_ids:
            raise ValueError(
                "Cannot render phase map: PhaseMapData.phases is empty. "
                "This usually means the xmap has no indexed pixels or "
                "phase extraction failed upstream."
            )

        cmap = ListedColormap(colors)
        boundaries = []
        for i, pid in enumerate(phase_ids):
            boundaries.append(pid - 0.5)
        boundaries.append(phase_ids[-1] + 0.5)
        norm = BoundaryNorm(boundaries, len(colors))

        ax.imshow(
            data.phase_map_2d,
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            origin='upper',
            aspect='equal',
        )

    ax.set_axis_off()
    ax.set_title(data.title, fontsize=12, pad=8)

    # Add scalebar
    if scalebar_config.enabled and data.step_x > 0:
        scalebar = ScaleBar(
            data.step_x,
            'um',
            length_fraction=scalebar_config.length_fraction,
            location=scalebar_config.location,
            color=scalebar_config.color,
            box_color=scalebar_config.box_color,
            box_alpha=scalebar_config.box_alpha,
            font_properties={'size': scalebar_config.font_size},
            border_pad=scalebar_config.border_pad,
            sep=scalebar_config.sep,
        )
        ax.add_artist(scalebar)

    # Add legend (only for discrete phase maps, not IPF)
    if show_legend and not is_rgb:
        import matplotlib.patches as mpatches
        legend_handles = []
        for phase in data.phases:
            patch = mpatches.Patch(color=phase.color, label=phase.name)
            legend_handles.append(patch)
        ax.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(data.phases), 4),
            fontsize=9,
            frameon=True,
            facecolor='#282a36',
            edgecolor='#6272a4',
            labelcolor='white',
        )

    fig.tight_layout()
    return fig


def save_phase_map(
    fig: Figure,
    output_path: str,
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """Save a rendered phase map figure to disk.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure from render_phase_map().
    output_path : str
        Output file path (.png, .tiff, .pdf, .svg).
    dpi : int
        Export resolution.
    transparent : bool
        Transparent background.

    Returns
    -------
    Path
        Path to saved file.
    """
    output = Path(output_path)
    # Make sure the parent directory exists. Without this, saving to a
    # nested path the user typed in (e.g. C:/exports/2026-04/phasemap.png)
    # raised FileNotFoundError when the date subdirectory hadn't been
    # created yet.
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output),
        dpi=dpi,
        bbox_inches='tight',
        facecolor=fig.get_facecolor() if not transparent else 'none',
        edgecolor='none',
        transparent=transparent,
    )
    logger.info(f"Phase map saved to {output}")
    return output
