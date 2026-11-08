"""
Grain Analysis — Grain reconstruction and grain size analysis.

This module provides:
- reconstruct_grains(): grains of an indexed square map, narrow not-indexed areas closed
- analyze_grain_size(): ECD, MaxDim, Aspect Ratio, area-weighted statistics
- calculate_grain_properties(): size, mean orientation, GOS and shape per grain

Grains are found on the cells of the measurements (analysis/grain_reconstruction.py):
neighbours of the same phase below the disorientation threshold are one grain,
narrow not-indexed areas are covered by the surrounding grains, and
measurements in grains below ``min_pixels`` are treated as not indexed.
Sizes and shape (area, perimeters, aspect ratio) are read from the grain cells,
so a grain's area includes the covered not-indexed pixels inside it; orientation
values (mean orientation, GOS) and boundary angles use the measurements only.

Method: Bachmann, Hielscher, Schaeben, Ultramicroscopy 111 (2011) 1720-1733,
doi:10.1016/j.ultramic.2011.08.002.
"""

from dataclasses import dataclass
import logging
import weakref
import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

try:
    from orix.quaternion import Orientation
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

from analysis.ebsd_dataset import EBSDDataset, GrainSet
from analysis.grain_cells import boundary_lengths, build_cells, grain_moments
from analysis.grain_reconstruction import grid_arrays, phase_ops, segment
from analysis.orientation_math import disorientation_angle, label_means, label_spread


# Which grains the grain-size statistics leave out.
#   matlab_script  the rule of matlab_testskripts/GrainSizeAnalysis_mtex6.m (default)
#   mtex           MTEX ``grains.isBoundary``
#   outer_row      grains with a pixel in the first or last row or column
#   keep_all       none
EDGE_RULES = ("matlab_script", "mtex", "outer_row", "keep_all")
# What the ECD (and its area weight) is made from: the area of the grain's cells, or the number of its
# measured pixels times the step squared (the script's ``grainSize * step^2``).
ECD_BASES = ("cell_area", "measured_pixels")
# matlab_script and mtex keep every grain when fewer than this many, or this fraction, would remain.
EDGE_MIN_GRAINS = 50
EDGE_MIN_FRACTION = 0.15
# ``stats_phase`` besides None (all phases) and a phase id: the phase with the most measured pixels
MOST_MEASURED = "most_measured"


@dataclass
class GrainSizeResult:
    """
    Grain size analysis results.

    Attributes:
        avg_maxDimX: Mean maximum X dimension (µm)
        std_maxDimX: Std deviation of maxDimX
        avg_maxDimY: Mean maximum Y dimension (µm)
        std_maxDimY: Std deviation of maxDimY
        avg_ECD: Mean equivalent circular diameter (µm)
        std_ECD: Std deviation of ECD
        avg_AR: Mean aspect ratio: long over short principal axis of the grain's shape, >= 1
        std_AR: Std deviation of aspect ratio
        area_weighted_ECD: Area-weighted mean ECD (µm)

        Distributions (for histograms):
        ecd_per_grain: ECD per grain (µm), length n_grains
        maxDimX_per_grain: MaxDimX per grain (µm)
        maxDimY_per_grain: MaxDimY per grain (µm)
        ar_per_grain: Aspect ratio per grain

        Settings and counts (what the numbers above are made of):
        edge_rule: The edge-grain rule applied (``keep_all`` when edge removal was switched off)
        ecd_basis: ``cell_area`` or ``measured_pixels``
        stats_phase: The phase id the statistics are made of (``most_measured`` resolved), None = all phases
        stats_phase_name: The name of that phase, None for all phases
        n_grains_total: Grains before any removal (of the phase, when one is chosen)
        n_edge_grains: Edge grains the rule found (before the safeguard)
        n_grains_used: Grains the statistics are made of
        edge_safeguard_applied: The rule found edge grains but removing them would have left too few,
            so all grains stayed

    Reference: the MTEX-equivalence design notes §2.2
    """
    avg_maxDimX: float
    std_maxDimX: float
    avg_maxDimY: float
    std_maxDimY: float
    avg_ECD: float
    std_ECD: float
    avg_AR: float
    std_AR: float
    area_weighted_ECD: float

    # Distributions
    ecd_per_grain: np.ndarray
    maxDimX_per_grain: np.ndarray
    maxDimY_per_grain: np.ndarray
    ar_per_grain: np.ndarray

    edge_rule: str = "matlab_script"
    ecd_basis: str = "cell_area"
    n_grains_total: int = 0
    n_edge_grains: int = 0
    n_grains_used: int = 0
    edge_safeguard_applied: bool = False
    stats_phase: "int | None" = None
    stats_phase_name: "str | None" = None


def reconstruct_grains(
    dataset: EBSDDataset,
    angle_threshold_deg: float = 5.0,
    min_pixels: int = 3,
    alpha: float = 3.1,
    fill: bool = True,
) -> GrainSet:
    """Grains of an indexed square map, narrow not-indexed areas closed.

    Neighbours of the same phase with a disorientation below
    ``angle_threshold_deg`` form one grain; different phases are always
    separated; not-indexed areas narrower than about ``2 * alpha`` pixels
    (holes, preparation scratches) are covered by the surrounding grains;
    measurements in grains below ``min_pixels`` are treated as not indexed and
    the map is segmented again. ``fill`` gives the covered pixels an
    orientation from their own grain (display only; no statistic uses it).

    Method: Bachmann, Hielscher, Schaeben, Ultramicroscopy 111 (2011)
    1720-1733, doi:10.1016/j.ultramic.2011.08.002, on the pixel grid.
    """
    if not angle_threshold_deg > 0:
        raise ValueError(f"angle_threshold_deg must be > 0, got {angle_threshold_deg}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if int(min_pixels) < 1:
        raise ValueError(f"min_pixels must be >= 1, got {min_pixels}")
    quats, phase = grid_arrays(dataset)
    ops = phase_ops(dataset.xmap)
    seg = segment(quats, phase, ops, np.deg2rad(angle_threshold_deg),
                  min_pixels=int(min_pixels), alpha=float(alpha))
    grains = GrainSet(grain_ids=seg.grain_ids, n_grains=seg.n_grains)
    grains.closed_mask = seg.closed
    grains.culled_mask = seg.culled
    grains.dropped_mask = seg.dropped
    grains.cells = seg.cells
    grains.grain_phase = seg.grain_phase
    grains.alpha = float(alpha)
    if seg.n_grains > 0:
        grains = calculate_grain_properties(dataset, grains)
        if fill:
            from analysis.grain_fill import fill_closed_pixels
            grains.filled_rotations, grains.filled_mask = fill_closed_pixels(
                quats, seg.grain_ids, seg.measured, seg.grain_phase, ops)
    return grains


def _phase_of_grains(gid, phase, measured, n):
    out = np.full(n, -1, dtype=np.int64)
    idx = np.flatnonzero(measured & (gid > 0))
    out[gid[idx] - 1] = phase[idx]
    return out


def _aspect_ratio(moments):
    """Long over short principal axis of the grain cells, from the central second moments.

    Hypot form of sqrt(lmax / lmin): the eigenvalues are mean +- hypot((cxx - cyy) / 2, cxy), and the
    quotient of the plain eigenvalue formula would lose digits to cancellation for slender grains.
    """
    cxx, cyy, cxy = moments["cxx"], moments["cyy"], moments["cxy"]
    mean = (cxx + cyy) / 2.0
    r = np.hypot((cxx - cyy) / 2.0, cxy)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((mean + r) / (mean - r))


def _boundary_segments(cells, gid, measured, grain_phase, q, ops, step):
    """Grain boundary segments: pairs of measurements of different grains of one phase whose cells share
    an edge. Returns (pairs (K, 2) flat indices a < b, angle in degrees (K,), length in um (K,)).

    Several edges of one pair are one segment (their lengths are summed); a pair of different phases has
    no misorientation and is left out.
    """
    ec = cells.edge_cells
    ok = (ec[:, 0] >= 0) & (ec[:, 1] >= 0)
    ok[ok] = measured[ec[ok, 0]] & measured[ec[ok, 1]] & (gid[ec[ok, 0]] != gid[ec[ok, 1]])
    pairs, length = ec[ok], cells.edge_length[ok]
    if len(pairs):
        same_phase = grain_phase[gid[pairs[:, 0]] - 1] == grain_phase[gid[pairs[:, 1]] - 1]
        pairs, length = pairs[same_phase], length[same_phase]
    if len(pairs) == 0:
        return pairs.reshape(0, 2), np.zeros(0), np.zeros(0)
    pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    length = np.bincount(inverse.reshape(-1), weights=length, minlength=len(pairs))
    angle = np.zeros(len(pairs))
    phase_of_pair = grain_phase[gid[pairs[:, 0]] - 1]
    for pid in np.unique(phase_of_pair):
        s = phase_of_pair == pid
        angle[s] = np.rad2deg(disorientation_angle(q[pairs[s, 0]], q[pairs[s, 1]], ops[int(pid)]))
    return pairs, angle, length * step


def calculate_grain_properties(dataset: EBSDDataset, grains: GrainSet) -> GrainSet:
    """Sizes and shape from the grain cells, orientation values from the measurements.

    ``grains.cells`` (set by ``reconstruct_grains``) holds the cells of the grains' last segmentation
    pass: area, perimeters and the aspect ratio are read from them, so a grain's area includes the
    covered not-indexed pixels it encloses. A GrainSet built by hand has no cells; every pixel then
    owns its square. ``grain_size`` is the number of measurements, ``n_closed`` the pixels of the grain
    that are not measurements. Boundaries are the cell edges between measurements of different grains
    of one phase.
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix required for grain properties")
    quats, phase = grid_arrays(dataset)
    ny, nx = grains.grain_ids.shape
    n = int(grains.n_grains)
    gid = grains.grain_ids.reshape(-1).astype(np.int64)
    if gid.size and int(gid.max()) > n:
        raise ValueError("grain_ids hold a label above n_grains")
    closed = (grains.closed_mask.reshape(-1) if grains.closed_mask is not None
              else np.zeros(ny * nx, dtype=bool))
    q = quats.reshape(-1, 4)
    ph = phase.reshape(-1)
    measured = (gid > 0) & ~closed & (ph >= 0) & np.isfinite(q).all(axis=1)
    grain_phase = (grains.grain_phase if grains.grain_phase is not None
                   else _phase_of_grains(gid, ph, measured, n))
    ops = phase_ops(dataset.xmap)

    cells = grains.cells if grains.cells is not None else build_cells(np.ones((ny, nx), dtype=bool))
    if cells.shape != (ny, nx):
        raise ValueError(f"the grain cells are for a {cells.shape} map, the grains for {(ny, nx)}")
    # Only generator pixels are read: there a label is the grain of a measurement, and 0 on the open
    # not-indexed pixels, whose grain id is 0 as well (pixels covered by the cells around them are no
    # generators and are never read).
    moments = grain_moments(cells, gid, n)
    outer, with_inclusions = boundary_lengths(cells, gid, n)
    grain_size = np.bincount(gid[measured], minlength=n + 1)[1:]
    n_closed = np.bincount(gid[(gid > 0) & ~measured], minlength=n + 1)[1:]
    area = moments["area"] * dataset.step_size ** 2
    ecd = 2.0 * np.sqrt(area / np.pi)

    means = np.full((n, 4), np.nan)
    gos = np.full(n, np.nan)
    for pid in np.unique(grain_phase[grain_phase >= 0]):
        of_phase = grain_phase == pid
        px = measured & of_phase[np.maximum(gid, 1) - 1]
        m = label_means(q[px], gid[px], n, ops[int(pid)])
        s = label_spread(q[px], gid[px], m, ops[int(pid)])
        means[of_phase] = m[of_phase]
        gos[of_phase] = s[of_phase]

    pairs, angle, length = _boundary_segments(cells, gid, measured, grain_phase, q, ops, dataset.step_size)

    grains.cells = cells
    grains.grain_phase = grain_phase
    grains.mean_orientations = Orientation(means, symmetry=dataset.phase.point_group)
    grains.grain_size = grain_size
    grains.n_closed = n_closed
    grains.area = area
    grains.ecd = ecd
    grains.gos = gos
    grains.aspect_ratio = _aspect_ratio(moments)
    grains.perimeter = outer * dataset.step_size
    grains.perimeter_with_inclusions = with_inclusions * dataset.step_size
    grains.boundary_pairs = pairs
    grains.boundary_type = angle
    grains.boundary_length = length
    return grains


def _check_choice(name, value, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def edge_safeguard_triggered(edge_rule: str, n_total: int, n_edge: int,
                             edge_tolerance_fraction: float = 0.85) -> bool:
    """True when the ``n_edge`` edge grains are not removed because that would remove too many.

    ``matlab_script`` and ``mtex``: fewer than 50 grains, or less than 15 % of all grains, would remain
    (the script's safeguard). ``outer_row``: more than ``edge_tolerance_fraction`` of all grains are edge
    grains. Without edge grains, and for ``keep_all``, nothing is removed and nothing is guarded.
    """
    _check_choice("edge_rule", edge_rule, EDGE_RULES)
    if edge_rule == "keep_all" or n_edge <= 0 or n_total <= 0:
        return False
    if edge_rule == "outer_row":
        return n_edge / n_total > edge_tolerance_fraction
    remaining = n_total - n_edge
    return remaining < EDGE_MIN_GRAINS or remaining / n_total < EDGE_MIN_FRACTION


def grain_ecd_and_area(dataset: EBSDDataset, grains: GrainSet, ecd_basis: str = "cell_area"):
    """(ECD, area) per grain in um and um^2 on the chosen basis: ``cell_area`` is what the GrainSet holds
    (the area of the grain's cells), ``measured_pixels`` is ``grain_size * step**2``."""
    _check_choice("ecd_basis", ecd_basis, ECD_BASES)
    if ecd_basis == "cell_area":
        return grains.ecd, grains.area
    if grains.grain_size is None:
        raise ValueError("the measured_pixels ECD basis needs grain_size, which this GrainSet does not have")
    area = np.asarray(grains.grain_size, dtype=float) * float(dataset.step_size) ** 2
    return 2.0 * np.sqrt(area / np.pi), area


def _measured_mask(dataset: EBSDDataset, grains: GrainSet, phase=None):
    """Flat mask of the measurements the grains are made of (indexed, in a grain, not closed); with
    ``phase`` only those of that phase id."""
    ny, nx = grains.grain_ids.shape
    if tuple(dataset.shape) != (ny, nx):
        raise ValueError(f"the dataset is {tuple(dataset.shape)}, the grains are for a {(ny, nx)} map")
    quats, phase_map = grid_arrays(dataset)
    gid = grains.grain_ids.reshape(-1)
    closed = (grains.closed_mask.reshape(-1) if grains.closed_mask is not None
              else np.zeros(ny * nx, dtype=bool))
    ph = phase_map.reshape(-1)
    measured = (gid > 0) & ~closed & (ph >= 0) & np.isfinite(quats.reshape(-1, 4)).all(axis=1)
    if phase is not None:
        measured &= ph == int(phase)
    return measured


def phase_ids(dataset: EBSDDataset) -> list:
    """The phase ids of the map (not-indexed, id < 0, left out), ascending."""
    return sorted(int(i) for i in dataset.xmap.phases.ids if int(i) >= 0)


def phase_name(dataset: EBSDDataset, phase_id: int) -> str:
    """Name of a phase of the map (``phase <id>`` when it has none)."""
    return str(getattr(dataset.xmap.phases[int(phase_id)], "name", "") or "") or f"phase {int(phase_id)}"


def resolve_stats_phase(dataset: EBSDDataset, grains: GrainSet, stats_phase):
    """``stats_phase`` as a phase id or None: None = all phases, an int = that phase,
    ``"most_measured"`` = the phase with the most measured pixels (the lowest id on a tie).

    Raises ValueError for anything else and for an id the map does not have; the message lists the ids."""
    if stats_phase is None:
        return None
    known = phase_ids(dataset)
    if isinstance(stats_phase, str):
        if stats_phase != MOST_MEASURED:
            raise ValueError(f"stats_phase must be None, a phase id {known} or {MOST_MEASURED!r}, got {stats_phase!r}")
        measured = _measured_mask(dataset, grains)
        _, phase = grid_arrays(dataset)
        counts = np.bincount(phase.reshape(-1)[measured], minlength=(max(known) + 1) if known else 1)
        counts = [(int(counts[i]) if i < len(counts) else 0, -i) for i in known]
        return -max(counts)[1] if counts else None
    if isinstance(stats_phase, (bool, np.bool_)) or not isinstance(stats_phase, (int, np.integer)):
        raise ValueError(f"stats_phase must be None, a phase id {known} or {MOST_MEASURED!r}, got {stats_phase!r}")
    if int(stats_phase) not in known:
        raise ValueError(f"stats_phase {int(stats_phase)} is not a phase of this map; the phase ids are {known}")
    return int(stats_phase)


def find_edge_grains(dataset: EBSDDataset, grains: GrainSet, edge_rule: str = "matlab_script",
                     phase=None) -> np.ndarray:
    """Boolean ``(n_grains,)``: which grains the ``edge_rule`` calls edge grains (grain ``i + 1`` at ``i``).

    ``matlab_script``: the extent is the smallest and largest row and column of the measured pixels; a
    measured pixel is an edge pixel when its row is <= min + 1 or >= max - 1, or its column likewise (one
    step of inset, so a band two pixels wide); the grains of the edge pixels are edge grains. This is
    ``GrainSizeAnalysis_mtex6.m``, which works on one phase's filled data (``ebsd_ph``); here the measured
    pixels of all phases are taken, or with ``phase`` those of that phase id only.

    ``mtex``: a grain is an edge grain when one of its measured cells has an edge to the exterior ring,
    or shares an edge with an open cell whose 4-connected region of open cells touches the map border.
    An open cell is the cell of a generator pixel without a grain that is not covered by the closing:
    an originally not-indexed pixel, or a measurement culled in the first ``min_pixels`` pass. A
    measurement still too small after the second pass (``grains.dropped_mask``) keeps its cell but
    is not open. This is MTEX ``grains.isBoundary`` as measured from its outputs: all grains agree on
    four real maps (78/78, 176/176, 76/76, 969/969), and any other reading of the open cells does not
    (without the culled ones 2 of 78 differ, with the finally dropped ones 5 of 969); it needs
    ``grains.cells``.

    ``outer_row``: grains with a pixel (closed pixels too) in the first or last row or column.

    ``keep_all``: none. The safeguards are not applied here (``edge_safeguard_triggered``).

    ``phase`` (a phase id): only grains of that phase are ever flagged, and the rules treat the other
    phases differently. ``matlab_script`` works on that phase's measured pixels alone, as the script's
    ``ebsd_ph`` does: the band is that of the phase's own extent, so a grain next to another phase can be
    an edge grain. ``mtex`` and ``outer_row`` look at the whole map: grains of other phases are solid
    neighbours, not open space, and the phase only selects which grains are flagged (and later counted),
    as MTEX ``grains(ph).isBoundary`` asks about edges to the outside or to not-indexed space. A grain
    that borders only another phase is not an edge grain under these two rules.
    """
    _check_choice("edge_rule", edge_rule, EDGE_RULES)
    n = int(grains.n_grains)
    edge = np.zeros(n, dtype=bool)
    if n == 0 or edge_rule == "keep_all":
        return edge
    gid_map = grains.grain_ids
    ny, nx = gid_map.shape

    if phase is not None and grains.grain_phase is None:
        raise ValueError("a phase filter needs grain_phase, which this GrainSet does not have")
    if edge_rule == "outer_row":
        ids = np.unique(np.concatenate([gid_map[0, :], gid_map[-1, :], gid_map[:, 0], gid_map[:, -1]]))
        edge[ids[(ids > 0) & (ids <= n)] - 1] = True
        if phase is not None:
            edge &= np.asarray(grains.grain_phase) == int(phase)
        return edge

    if edge_rule == "mtex" and grains.cells is None:
        raise ValueError("the mtex edge rule needs the grain cells (grains.cells), which this GrainSet "
                         "does not have; reconstruct the grains with reconstruct_grains first")
    measured_all = _measured_mask(dataset, grains)
    measured = _measured_mask(dataset, grains, phase) if edge_rule == "matlab_script" else measured_all
    gid = gid_map.reshape(-1).astype(np.int64)

    if edge_rule == "matlab_script":
        rows, cols = np.divmod(np.flatnonzero(measured), nx)
        if rows.size == 0:
            return edge
        band = ((rows <= rows.min() + 1) | (rows >= rows.max() - 1) |
                (cols <= cols.min() + 1) | (cols >= cols.max() - 1))
        flagged = gid_map[rows[band], cols[band]]
    else:
        cells = grains.cells
        if cells.shape != (ny, nx):
            raise ValueError(f"the grain cells are for a {cells.shape} map, the grains for {(ny, nx)}")
        a, b = cells.edge_cells[:, 0], cells.edge_cells[:, 1]
        # (a) an edge to the exterior ring (-1)
        one_out = (a < 0) != (b < 0)
        own = np.where(a[one_out] >= 0, a[one_out], b[one_out])
        flagged = [gid[own[measured[own]]]]
        # (b) an edge to an open cell (no grain, not finally dropped) whose region of open cells reaches
        # the map border
        is_open = cells.generator.reshape(-1) & (gid == 0)
        if grains.dropped_mask is not None:
            is_open &= ~grains.dropped_mask.reshape(-1)
        labels, _ = ndi.label(is_open.reshape(ny, nx))          # 4-connected (cross structure)
        at_border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        open_border = np.isin(labels.reshape(-1), at_border[at_border > 0])
        both = (a >= 0) & (b >= 0)
        p, q = a[both], b[both]
        flagged.append(gid[p[measured[p] & open_border[q]]])
        flagged.append(gid[q[measured[q] & open_border[p]]])
        flagged = np.concatenate(flagged)
    ids = np.unique(flagged)
    edge[ids[(ids > 0) & (ids <= n)] - 1] = True
    if phase is not None:
        edge &= np.asarray(grains.grain_phase) == int(phase)
    return edge


def _sample_std(values) -> float:
    """Standard deviation with divisor N - 1 (MATLAB ``std``); 0.0 for fewer than two values."""
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


# The grains object and key of the last "safeguard applied" warning: the statistics are recomputed by
# the ECD map and the Excel export, the warning is said once per change of grains or settings.
_safeguard_warned = {"grains": None, "key": None}


def _warn_safeguard_once(grains, key, message) -> None:
    ref = _safeguard_warned["grains"]
    if ref is not None and ref() is grains and _safeguard_warned["key"] == key:
        return
    _safeguard_warned["grains"] = weakref.ref(grains)
    _safeguard_warned["key"] = key
    logger.warning(message)


def grain_size_summary(result: GrainSizeResult) -> dict:
    """Medians of a result, of the chosen ECD basis, over the grains used (0.0 without grains).

    The one place the page, the reconstruct task result, the route and the Excel sheet take them from.
    """
    ecd, ar = result.ecd_per_grain, result.ar_per_grain
    return {
        "median_ecd": float(np.median(ecd)) if len(ecd) else 0.0,
        "median_aspect_ratio": float(np.median(ar)) if len(ar) else 0.0,
    }


def grain_size_basic_info(result: GrainSizeResult) -> dict:
    """The keys of sheet 1 of the Excel file that come from a grain-size result: counts, rule, basis,
    phase, the spreads and the weighted ECD of the script, medians and grains above 5/10/15 um (of the
    chosen ECD basis). Route export and batch both use it, so they cannot differ."""
    ecd = np.asarray(result.ecd_per_grain)
    medians = grain_size_summary(result)
    return {
        "grain_edge_rule": result.edge_rule,
        "grain_ecd_basis": result.ecd_basis,
        "n_grains_total": int(result.n_grains_total),
        "n_edge_grains": int(result.n_edge_grains),
        "n_grains_used": int(result.n_grains_used),
        "edge_safeguard_applied": bool(result.edge_safeguard_applied),
        "grain_stats_phase": result.stats_phase_name if result.stats_phase is not None else "all phases",
        "mean_MaxDimX_um": float(result.avg_maxDimX),
        "mean_MaxDimY_um": float(result.avg_maxDimY),
        "mean_AR": float(result.avg_AR),
        "std_maxDimX": float(result.std_maxDimX),
        "std_maxDimY": float(result.std_maxDimY),
        "std_AR": float(result.std_AR),
        "area_weighted_ECD": float(result.area_weighted_ECD),
        "median_ECD_um": medians["median_ecd"],
        "median_AR": medians["median_aspect_ratio"],
        "grains_above_5um": int(np.sum(ecd > 5.0)),
        "grains_above_10um": int(np.sum(ecd > 10.0)),
        "grains_above_15um": int(np.sum(ecd > 15.0)),
    }


def analyze_grain_size(
    dataset: EBSDDataset,
    grains: GrainSet,
    remove_edge_grains: bool = True,
    edge_tolerance_fraction: float = 0.85,
    edge_rule: str = "matlab_script",
    ecd_basis: str = "cell_area",
    stats_phase=None,
) -> GrainSizeResult:
    """
    Analyze grain size: ECD, MaxDim, Aspect Ratio, area-weighted statistics.

    Workflow:
    1. Optionally restrict to the grains of one phase (``stats_phase``)
    2. Optionally remove edge grains by ``edge_rule`` (with the rule's safeguard)
    3. Calculate MaxDimX, MaxDimY per grain
    4. Compute summary statistics (mean, std with divisor N - 1 like MATLAB ``std``; 0.0 for one grain)
    5. Compute area-weighted ECD

    Args:
        dataset: EBSDDataset with step_size
        grains: GrainSet with calculated properties
        remove_edge_grains: False means ``edge_rule="keep_all"`` (default True)
        edge_tolerance_fraction: ``outer_row`` only: if more than this fraction of the grains are edge
            grains, none are removed (default 0.85)
        edge_rule: ``matlab_script`` (default), ``mtex``, ``outer_row`` or ``keep_all``, see
            ``find_edge_grains``; ``matlab_script`` and ``mtex`` keep all grains when fewer than 50, or
            less than 15 %, would remain
        ecd_basis: ``cell_area`` (default; the area of the grain's cells, MTEX ``equivalentRadius``) or
            ``measured_pixels`` (``grain_size * step**2``, as the MATLAB script); the choice applies to
            the ECD and to the area weights of the weighted ECD
        stats_phase: None (default) = all phases; a phase id = only the grains of that phase;
            ``"most_measured"`` = the phase with the most measured pixels. The choice is made before the
            edge rule: band extent, safeguard counts and every value refer to the grains of that phase
            only, as the script does with particle removal (``grains(ph)``)

    Returns:
        GrainSizeResult with statistics, distributions, the rule and basis used and the counts of
        grains (total, edge, used) and whether the safeguard kept all grains

    Raises:
        ValueError: unknown ``edge_rule`` or ``ecd_basis``; ``mtex`` on a GrainSet without cells;
            ``measured_pixels`` on a GrainSet without ``grain_size``; an unknown ``stats_phase`` (the
            message lists the phase ids); a ``stats_phase`` on a GrainSet without ``grain_phase``

    Reference: the MATLAB reference scripts in ``matlab_testskripts/`` and the grain statistics of MTEX.
    """
    _check_choice("edge_rule", edge_rule, EDGE_RULES)
    _check_choice("ecd_basis", ecd_basis, ECD_BASES)
    rule = edge_rule if remove_edge_grains else "keep_all"
    phase = resolve_stats_phase(dataset, grains, stats_phase)
    phase_label = phase_name(dataset, phase) if phase is not None else None
    n_all = int(grains.n_grains)
    if phase is not None and n_all > 0 and grains.grain_phase is None:
        raise ValueError("stats_phase needs grain_phase, which this GrainSet does not have")
    in_phase = (np.ones(n_all, dtype=bool) if phase is None
                else np.asarray(grains.grain_phase) == phase)
    n_total = int(in_phase.sum())

    # Handle zero-grain case gracefully (e.g. fully random data with no boundaries)
    if n_total == 0 or grains.ecd is None or grains.area is None:
        empty = np.array([], dtype=float)
        return GrainSizeResult(
            avg_maxDimX=0.0, std_maxDimX=0.0,
            avg_maxDimY=0.0, std_maxDimY=0.0,
            avg_ECD=0.0, std_ECD=0.0,
            avg_AR=0.0, std_AR=0.0,
            area_weighted_ECD=0.0,
            ecd_per_grain=empty,
            maxDimX_per_grain=empty,
            maxDimY_per_grain=empty,
            ar_per_grain=empty,
            edge_rule=rule, ecd_basis=ecd_basis, n_grains_total=n_total,
            stats_phase=phase, stats_phase_name=phase_label,
        )

    ecd_all, area_all = grain_ecd_and_area(dataset, grains, ecd_basis)

    # Identify edge grains (of the chosen phase)
    edge = find_edge_grains(dataset, grains, rule, phase=phase)
    n_edge = int(edge.sum())
    safeguard = edge_safeguard_triggered(rule, n_total, n_edge, edge_tolerance_fraction)
    if safeguard:
        _warn_safeguard_once(
            grains, (rule, phase, n_total, n_edge),
            "edge rule %s%s: %d of %d grains are edge grains; removing them would leave too few, "
            "keeping all grains." % (rule, "" if phase is None else f" (phase {phase})", n_edge, n_total))
        edge = np.zeros(n_all, dtype=bool)
    else:
        _safeguard_warned["grains"] = None      # the next time it applies, it is said again
    valid_mask = in_phase & ~edge

    # Calculate MaxDimX, MaxDimY per grain — vectorized via find_objects
    from scipy.ndimage import find_objects
    maxDimX = np.zeros(grains.n_grains)
    maxDimY = np.zeros(grains.n_grains)

    slices = find_objects(grains.grain_ids)
    for i, sl in enumerate(slices):
        if sl is not None:
            maxDimX[i] = (sl[1].stop - sl[1].start) * dataset.step_size
            maxDimY[i] = (sl[0].stop - sl[0].start) * dataset.step_size

    # Filter to valid grains
    ecd_valid = ecd_all[valid_mask]
    maxDimX_valid = maxDimX[valid_mask]
    maxDimY_valid = maxDimY[valid_mask]
    ar_valid = grains.aspect_ratio[valid_mask]
    area_valid = area_all[valid_mask]

    # Summary statistics; std is the sample standard deviation (divisor N - 1), as MATLAB ``std``
    avg_ecd = np.mean(ecd_valid) if len(ecd_valid) > 0 else 0.0
    std_ecd = _sample_std(ecd_valid)

    avg_maxDimX = np.mean(maxDimX_valid) if len(maxDimX_valid) > 0 else 0.0
    std_maxDimX = _sample_std(maxDimX_valid)

    avg_maxDimY = np.mean(maxDimY_valid) if len(maxDimY_valid) > 0 else 0.0
    std_maxDimY = _sample_std(maxDimY_valid)

    avg_ar = np.mean(ar_valid) if len(ar_valid) > 0 else 0.0
    std_ar = _sample_std(ar_valid)

    # Area-weighted ECD
    if len(ecd_valid) > 0 and area_valid.sum() > 0:
        area_weighted_ecd = np.sum(ecd_valid * area_valid) / np.sum(area_valid)
    else:
        area_weighted_ecd = 0.0

    return GrainSizeResult(
        avg_maxDimX=avg_maxDimX,
        std_maxDimX=std_maxDimX,
        avg_maxDimY=avg_maxDimY,
        std_maxDimY=std_maxDimY,
        avg_ECD=avg_ecd,
        std_ECD=std_ecd,
        avg_AR=avg_ar,
        std_AR=std_ar,
        area_weighted_ECD=area_weighted_ecd,
        ecd_per_grain=ecd_valid,
        maxDimX_per_grain=maxDimX_valid,
        maxDimY_per_grain=maxDimY_valid,
        ar_per_grain=ar_valid,
        edge_rule=rule,
        ecd_basis=ecd_basis,
        n_grains_total=n_total,
        n_edge_grains=n_edge,
        n_grains_used=int(valid_mask.sum()),
        edge_safeguard_applied=bool(safeguard),
        stats_phase=phase,
        stats_phase_name=phase_label,
    )
