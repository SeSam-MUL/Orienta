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


def compute_ipf_colors(xmap, direction_str: str = "Z", r_user=None) -> np.ndarray:
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
        if r_user is not None and not np.allclose(
            np.asarray(r_user.data), np.asarray(Rotation.identity().data)
        ):
            # Apply the sample-frame rotation (right-multiply — same convention
            # as orientation_frame.to_vendor_export_frame) WITHOUT mutating the
            # stored xmap. Re-wrap as Orientation with the phase symmetry so the
            # colour key reduces into the fundamental zone correctly.
            rot_disp = subset.rotations * r_user
            oris = Orientation(rot_disp, symmetry=pg)
        else:
            oris = subset.orientations
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
