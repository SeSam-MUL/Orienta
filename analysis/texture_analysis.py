"""
Texture Analysis Module for EBSD Data.

Implements:
- The ideal orientations of the texture components as the MTEX script builds them (``ideal_orientation``,
  ``component_variants``): {hkl}<uvw> with the plane normal along ND and a crystal equivalent of uvw that is
  perpendicular to it along RD, times the specimen symmetry mmm
- Volume fraction calculation for texture components (14 FCC rolling components), counted as the script counts:
  per component the SUM over its distinct variants of the share of pixels within the tolerance of that variant
- The angle of every indexed pixel to every variant of every component, computed once per dataset
  (``get_component_angles``) and shared by the component fractions, the RX / substructure split and
  the component map
- Surface plane fractions for {111}, {100}, {110}: the script's number (one direction per pixel) and the share of
  pixels with any plane of the family in the surface

- Texture index and entropy of the 5 degree kernel density of the orientations (``analysis.texture_odf``), the
  script's ``norm(odf)^2`` and ``entropy(odf)``

Everything here counts the pixels of ONE phase, the texture phase: the phase with the most indexed pixels of the map
(default) or a phase id the caller names (``resolve_texture_phase``; it must be cubic, the components are). The script
does the same (``oriR = ebsd(ph).orientations``, ``ph = 'aluminium'`` or ``'indexed'``); pixels of other phases and
not-indexed pixels are neither counted nor in a denominator.

Reference: MATLAB/MTEX TextureAnalysis_mtex6.m lines 153-523
Architecture: the MTEX-equivalence design notes §2.4
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from orix.quaternion import Misorientation, Orientation
    from orix.vector import Miller, Vector3d
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

# Specimen symmetry mmm of a rolled sheet (RD = x, TD = y, ND = z): its proper rotations, as orix quaternions
# (identity and the three two-fold rotations). The improper elements of mmm only repeat these modulo the inversion
# every centrosymmetric crystal group contains. For a crystal group without inversion (-43m, 6mm, ...) the improper
# elements might give further variants in MTEX; only the proper elements are used here (no MTEX reference for it).
SPECIMEN_OPERATORS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# two orientations closer than this (rad) are one variant (MTEX ``unique`` uses 1e-3 rad; the variants of the
# components in use are several degrees apart)
VARIANT_TOLERANCE = 1e-6
# in the component map a later component must be closer than the earlier one by more than this (rad) to take a pixel
MAP_TIE_MARGIN = 1e-9


MOST_MEASURED = "most_measured"
# Laue classes of the cubic crystal system: the texture components (FCC / BCC presets) are cubic
CUBIC_LAUE_CLASSES = ("m-3m", "m-3")


def _indexed_phase_ids(dataset) -> List[int]:
    """The phase ids of the map (not-indexed, id < 0, left out), ascending."""
    return sorted(int(i) for i in np.asarray(dataset.xmap.phases.ids).reshape(-1) if int(i) >= 0)


def _phase_label(dataset, phase_id: int) -> str:
    name = str(getattr(dataset.xmap.phases[int(phase_id)], "name", "") or "")
    return name or f"phase {int(phase_id)}"


def _laue_class(dataset, phase_id: int) -> Optional[str]:
    """Name of the Laue class of a phase ('m-3m', 'm-3', '6/mmm', ...), None when it has no point group."""
    point_group = _phase_object(dataset, phase_id).point_group
    if point_group is None:
        return None
    try:
        return str(point_group.laue.name)
    except Exception:
        return str(point_group.name)


def _cubic_phases(dataset) -> List[str]:
    """``"<id> <name>"`` of the cubic phases of the map, for the messages."""
    return [f"{i} ({_phase_label(dataset, i)})" for i in _indexed_phase_ids(dataset)
            if _laue_class(dataset, i) in CUBIC_LAUE_CLASSES]


def _phase_id_of(dataset, texture_phase=MOST_MEASURED, what: str = "texture") -> int:
    """The id ``texture_phase`` names (``"most_measured"``: the phase with the most indexed pixels, the lowest id on a
    tie). ``ValueError`` for anything else and for an id the map does not have; the message lists the ids.
    The building blocks below (angles, fractions, map, split, planes) work on any phase with this; ``analyze_texture``
    and the callers above it go through ``resolve_texture_phase``, which also requires a cubic phase.
    ``what`` names the choice in the messages ("texture", or "rx" for the phase of the RX analysis, which uses the
    same rule)."""
    known = _indexed_phase_ids(dataset)
    if not known:
        raise ValueError("No indexed phases found in xmap")
    if isinstance(texture_phase, str) and texture_phase == MOST_MEASURED:
        ids = np.asarray(dataset.xmap.phase_id).reshape(-1)
        counts = [(int(np.count_nonzero(ids == i)), -i) for i in known]
        return -max(counts)[1]
    if isinstance(texture_phase, (bool, np.bool_)) or not isinstance(texture_phase, (int, np.integer)):
        raise ValueError(f"{what}_phase must be a phase id {known} or {MOST_MEASURED!r}, got {texture_phase!r}")
    pid = int(texture_phase)
    if pid not in known:
        raise ValueError(f"{what} phase {pid} is not a phase of this map; the phase ids are {known}")
    return pid


def resolve_texture_phase(dataset, texture_phase=MOST_MEASURED) -> int:
    """The id of the phase the texture is computed on.

    ``texture_phase``: ``"most_measured"`` (default) = the phase with the most INDEXED pixels of the map (the lowest
    id on a tie), or a phase id. This is the same rule as ``analysis.grain_analysis.resolve_stats_phase`` with
    ``"most_measured"``, but it counts the indexed pixels of the map and needs no grains: the grain statistics count
    the measured pixels after the grain reconstruction. The reference MATLAB script computes the texture on one phase
    (``ph = 'aluminium'``, the matrix, or ``'indexed'``).

    The components are cubic (FCC / BCC presets), so the chosen phase must have a cubic Laue class (m-3m or m-3).
    Raises ``ValueError`` for anything that is neither ``"most_measured"`` nor a phase id of the map (the message lists
    the ids), and for a phase that is not cubic (the message names the cubic phases of the map; for the default it says
    "choose a cubic phase").
    """
    default = isinstance(texture_phase, str) and texture_phase == MOST_MEASURED
    pid = _phase_id_of(dataset, texture_phase)
    laue = _laue_class(dataset, pid)
    if laue not in CUBIC_LAUE_CLASSES:
        cubic = _cubic_phases(dataset)
        listed = ("the cubic phases of this map are " + ", ".join(cubic)) if cubic else "this map has no cubic phase"
        if default:
            raise ValueError(
                f"The phase with the most indexed pixels, {pid} ({_phase_label(dataset, pid)}), is not cubic "
                f"(Laue class {laue}); the texture components are cubic. Choose a cubic phase: {listed}.")
        raise ValueError(
            f"Texture phase {pid} ({_phase_label(dataset, pid)}) is not cubic (Laue class {laue}); the texture "
            f"components are cubic. Choose a cubic phase: {listed}.")
    return pid


def cubic_phase_ids(dataset) -> List[int]:
    """The ids of the cubic phases of the map (Laue class m-3m or m-3): the phases a texture can be computed on."""
    return [i for i in _indexed_phase_ids(dataset) if _laue_class(dataset, i) in CUBIC_LAUE_CLASSES]


def texture_phase_info(dataset, texture_phase=MOST_MEASURED) -> Dict[str, Any]:
    """``{"id", "name"}`` of the texture phase ``texture_phase`` resolves to (see ``resolve_texture_phase``)."""
    pid = resolve_texture_phase(dataset, texture_phase)
    return {"id": pid, "name": _phase_label(dataset, pid)}


def _phase_object(dataset, phase_id: int):
    """The (frame-aware) phase object of a phase id."""
    from crystal_symmetry import frame_phase
    return frame_phase(dataset.xmap.phases[int(phase_id)])


def _phase_orientations(dataset, phase_id: int):
    """The orix Orientation of the pixels of a phase, with THAT phase's point group (``dataset.orientations`` has
    the symmetry of the first phase of the map)."""
    mask = np.asarray(dataset.xmap.phase_id) == int(phase_id)
    return Orientation(dataset.xmap.rotations[mask], symmetry=_phase_object(dataset, phase_id).point_group)


def _unit_miller(kind: str, indices, phase, name: str) -> np.ndarray:
    """Cartesian unit vector (crystal frame) of the plane normal (``hkl``) or direction (``uvw``)."""
    v = np.asarray(Miller(**{kind: list(indices)}, phase=phase).data, dtype=np.float64).reshape(-1)[:3]
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"{name}: {kind} {list(indices)} is the zero vector")
    return v / norm


def _matrices(quats: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) of unit quaternions (w, x, y, z)."""
    w, x, y, z = np.moveaxis(np.asarray(quats, dtype=np.float64).reshape(-1, 4), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def ideal_orientation(component: "TextureComponent", phase) -> np.ndarray:
    """The ideal orientation of a texture component as an orix quaternion (w, x, y, z), shape (4,).

    Miller components ``{hkl}<uvw>`` are built the way ``orientation.byMiller(hkl, uvw, CS)`` of the MTEX script
    builds them (measured with MTEX for Python on all twelve Miller components of the FCC preset): the plane
    normal ``hkl`` goes to ND (the specimen z axis), and of the crystal equivalents of ``uvw`` (the proper
    operators of the phase's point group applied to it) the one closest to perpendicular to ``hkl`` goes to RD
    (the specimen x axis), after removing its component along ``hkl``. ``uvw`` may be written non-perpendicular,
    like Brass {011}<211>: the equivalent <2 1 -1> is. When several equivalents are equally perpendicular the
    first one of the point group's operator list is taken; for the twelve components of the FCC preset every
    perpendicular choice gives the same component (same variants), which the tests check. Euler components use
    Bunge angles (degrees), as ``orientation.byEuler``.

    Raises:
        ValueError: a zero index vector, ``hkl`` parallel to every equivalent of ``uvw``, or a definition with
            neither Miller indices nor Euler angles
    """
    if component.hkl is not None and component.uvw is not None:
        ez = _unit_miller("hkl", component.hkl, phase, component.name)
        u = _unit_miller("uvw", component.uvw, phase, component.name)
        from analysis.orientation_math import proper_ops
        equivalents = np.einsum("nij,j->ni", _matrices(proper_ops(phase.point_group)), u)
        cos = np.abs(equivalents @ ez)
        best = equivalents[int(np.flatnonzero(cos <= cos.min() + 1e-9)[0])]       # first of the equally perpendicular
        x = best - (best @ ez) * ez
        norm = np.linalg.norm(x)
        if norm < 1e-9:
            raise ValueError(f"{component.name}: uvw is parallel to hkl in every symmetry equivalent")
        ex = x / norm
        # columns: the specimen axes (RD, TD, ND) written in the crystal frame
        from scipy.spatial.transform import Rotation as _Rot
        xyzw = _Rot.from_matrix(np.column_stack([ex, np.cross(ez, ex), ez])).as_quat()
        q = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    elif component.euler_deg is not None:
        q = np.asarray(Orientation.from_euler(np.radians(component.euler_deg),
                                              symmetry=phase.point_group).data, dtype=np.float64).reshape(4)
    else:
        raise ValueError(f"{component.name}: No valid definition (Miller or Euler)")
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def component_variants(component: "TextureComponent", phase) -> np.ndarray:
    """The distinct variants of a component under the specimen symmetry mmm: orix quaternions, shape (n, 4).

    ``unique(specimenSymmetry('mmm') * orientation)`` of the MTEX script, in orix's convention (an orix
    quaternion is the conjugate of the MTEX rotation, so the specimen operators act from the RIGHT and the
    crystal symmetry from the left): the ideal orientation times each of the four proper operators of mmm, with
    orientations that are equal modulo the crystal symmetry counted once. 1, 2 or 4 variants for the components
    of the FCC preset (``TextureComponent.symmetry_multiplicity``).
    """
    from analysis.orientation_math import disorientation_angle, proper_ops, qmul
    ideal = ideal_orientation(component, phase)
    ops = proper_ops(phase.point_group)
    kept: List[np.ndarray] = []
    for cand in qmul(ideal[None, :], SPECIMEN_OPERATORS):
        if all(disorientation_angle(cand[None], k[None], ops)[0] > VARIANT_TOLERANCE for k in kept):
            kept.append(cand if cand[0] >= 0 else -cand)
    return np.array(kept)


@dataclass
class TextureResult:
    """Results from texture analysis.

    Attributes:
        # Texture component volume fractions (14 components for FCC rolling)
        component_fractions: Dict[str, Optional[float]]  # Component name → volume fraction (%); None = could not be built

        # Surface plane fractions
        plane_111_fraction: float  # Fraction with {111} plane in surface (%)
        plane_100_fraction: float  # Fraction with {100} plane in surface (%)
        plane_110_fraction: float  # Fraction with {110} plane in surface (%)
        # (the script's number: ONE direction (h k l) of each pixel's stored orientation within the tolerance of Z)

        # Surface plane family fractions: any plane of the family {hkl} within the tolerance of Z
        plane_111_family_fraction: Optional[float]  # (%)
        plane_100_family_fraction: Optional[float]  # (%)
        plane_110_family_fraction: Optional[float]  # (%)

        # ODF-based metrics: of the 5 degree kernel density of the texture phase's orientations
        texture_index: Optional[float]  # Texture index, norm(odf)^2 (uniform = 1)
        entropy: Optional[float]  # Texture entropy, -integral f ln f (uniform = 0)

        # The phase everything above was computed on
        phase_id: Optional[int]
        phase_name: Optional[str]

        # Preprocessing
        specimen_rotation: Optional[np.ndarray]  # Applied specimen rotation (Euler angles, degrees)
        tolerance_angle: float  # Tolerance angle used for volume fractions (degrees)
    """
    component_fractions: Dict[str, Optional[float]]
    plane_111_fraction: Optional[float]
    plane_100_fraction: Optional[float]
    plane_110_fraction: Optional[float]
    texture_index: Optional[float] = None
    entropy: Optional[float] = None
    specimen_rotation: Optional[np.ndarray] = None
    tolerance_angle: float = 10.0
    plane_111_family_fraction: Optional[float] = None
    plane_100_family_fraction: Optional[float] = None
    plane_110_family_fraction: Optional[float] = None
    phase_id: Optional[int] = None
    phase_name: Optional[str] = None
    preset_name: Optional[str] = None


def calculate_volume_fraction(orientations: "Orientation", ideal_orientations: "Orientation",
                              tolerance_deg: float = 10.0,
                              symmetry=None) -> float:
    """Calculate volume fraction of orientations matching ideal component.

    For each EBSD orientation, finds the minimum misorientation angle to any of the
    ideal orientations (symmetry equivalents). Counts orientations within tolerance.

    The texture component fractions do NOT use this function: a pixel near two ideal orientations counts once
    here (union), while the script (and ``calculate_texture_component_fractions``) adds the share of each variant
    (``component_variants``) separately, so a pixel near two variants counts twice. Do not hand the variants of a
    component to this function and expect the script's number.

    Args:
        orientations: All EBSD orientations for the phase (flattened array)
        ideal_orientations: Ideal component orientations (symmetry equivalents)
        tolerance_deg: Maximum angular distance to count as belonging (degrees)
        symmetry: Crystal point group for misorientation calculation

    Returns:
        Volume fraction in percent (0-100)

    Reference: TextureAnalysis_mtex6.m lines 154-167 (EBSD-based volume fractions)
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    tolerance_rad = np.radians(tolerance_deg)

    n_ori = orientations.size
    if n_ori == 0:
        return 0.0

    min_angles = _min_angle_to_equivalents(orientations, ideal_orientations, symmetry)

    # Count orientations within tolerance
    within_tolerance = np.sum(min_angles < tolerance_rad)
    volume_fraction = (within_tolerance / n_ori) * 100.0

    return volume_fraction


def _min_angle_to_equivalents(orientations: "Orientation", ideal_orientations: "Orientation",
                              symmetry=None) -> np.ndarray:
    """Per orientation, the smallest misorientation angle (radians) to any of the ideal orientations.

    The angle is the smallest one among the crystal-symmetry equivalents of ``orientations * ~ideal`` (what
    orix' ``Misorientation(..., symmetry=(sym, sym)).reduce().angle`` returns, symmetry on both sides is one
    group acting on one side for the angle), computed with the vectorised orientation arithmetic of
    ``analysis.orientation_math``. Without ``symmetry`` the raw angle.
    """
    from analysis.orientation_math import min_disorientation_to_references, proper_ops

    q = np.asarray(getattr(orientations, "data", orientations), dtype=np.float64).reshape(-1, 4)
    ideals = np.asarray(getattr(ideal_orientations, "data", ideal_orientations), dtype=np.float64).reshape(-1, 4)
    ops = proper_ops(symmetry) if symmetry is not None else np.array([[1.0, 0.0, 0.0, 0.0]])
    return min_disorientation_to_references(q, ideals, ops)


@dataclass
class ComponentAngles:
    """The misorientation angle of every pixel of the components' phase to every variant of every texture component.

    ``variant_angles[k]`` (radians, float64, shape (n_variants, n_pixels)) holds, for component ``names[k]``, the
    angle of the i-th pixel of the phase (flat map order) to each of its variants
    (``component_variants``), the crystal symmetry of the phase applied. The angle to the component itself, the
    smallest over its variants, is ``min_angles[k]``: derived on demand (``min_angles`` stacks all components,
    ``component_min_angle(k)`` gives one), not stored a second time. A component that could not be built has NaN
    angles (one row), its fraction is 0 and it never claims a pixel; so does a pixel whose orientation is NaN. The
    angles do not depend on the tolerance, so every fraction, the RX / substructure split and the component map
    are read from them.
    """
    names: List[str]
    failed: List[bool]                # a component that could not be built
    key: tuple
    variant_angles: List[np.ndarray]  # per component (n_variants, n_pixels)
    n_pixels: int = 0

    def component_min_angle(self, k: int) -> np.ndarray:
        """(n_pixels,) the smallest angle of each pixel to the variants of component ``k`` (a view for one variant)."""
        rows = self.variant_angles[k]
        return rows[0] if len(rows) == 1 else rows.min(axis=0)

    @property
    def min_angles(self) -> np.ndarray:
        """(n_components, n_pixels) the smallest angle to each component, computed from ``variant_angles``."""
        if not self.names:
            return np.empty((0, self.n_pixels))
        return np.stack([self.component_min_angle(k) for k in range(len(self.names))])


def _phase_key(phase) -> tuple:
    try:
        return (str(phase.point_group.name), tuple(np.round(phase.structure.lattice.abcABG(), 9)))
    except Exception:
        return (str(getattr(phase, "point_group", None)),)


def _angles_key(dataset, components, phase, phase_mask) -> tuple:
    """What the angles depend on: the components, the phase, the pixels of the phase and their orientations."""
    comp_key = tuple(
        (name, None if c.hkl is None else tuple(c.hkl), None if c.uvw is None else tuple(c.uvw),
         None if c.euler_deg is None else tuple(float(e) for e in c.euler_deg))
        for name, c in components.items())
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(phase_mask).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(dataset.xmap.rotations.data, dtype=float)).tobytes())
    return (comp_key, _phase_key(phase), digest.hexdigest())


def get_component_angles(dataset, components: Dict[str, "TextureComponent"],
                         texture_phase=MOST_MEASURED) -> ComponentAngles:
    """The angles of the pixels of the texture phase to every variant of all components, once per dataset.

    The texture phase is the phase with the most indexed pixels or the phase id given
    (``resolve_texture_phase``); only its pixels are used (the script's ``ebsd(ph)``), so another phase or
    not-indexed pixels are neither compared nor counted.

    The result is kept on ``dataset`` (one entry) and reused as long as the components, the phase, the
    pixels of the phase and their orientations are the same; a quality filter or another orientation array
    computes it again.

    A NaN orientation gives NaN angles (the earlier orix path gave 0 there, which counted a NaN pixel as a
    perfect match for every component). Point groups that are improper without inversion (-43m, 6mm, ...)
    are accepted: their proper subgroup is used, orix' ``reduce()`` refused them.
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    pid = _phase_id_of(dataset, texture_phase)
    phase = _phase_object(dataset, pid)
    phase_mask = np.asarray(dataset.xmap.phase_id) == pid
    key = _angles_key(dataset, components, phase, phase_mask)
    cached = getattr(dataset, "_component_angles", None)
    if cached is not None and cached.key == key:
        return cached

    ori_flat = _phase_orientations(dataset, pid)

    n_idx = int(phase_mask.sum())
    names = list(components.keys())
    variant_angles: List[np.ndarray] = [np.full((1, n_idx), np.nan) for _ in names]
    failed = [True] * len(names)

    for k, (comp_name, comp) in enumerate(components.items()):
        # the distinct variants of the component under the specimen symmetry mmm (Miller indices or Euler angles)
        try:
            variants = component_variants(comp, phase)
        except Exception as e:
            logger.warning("Skipping texture component %s: %s", comp_name, e)
            continue

        # the angle to each variant (crystal symmetry applied for the smallest equivalent)
        rows = np.stack([_min_angle_to_equivalents(ori_flat, variants[j:j + 1], phase.point_group)
                         for j in range(len(variants))])
        variant_angles[k] = rows
        failed[k] = False

    result = ComponentAngles(names=names, failed=failed, key=key, variant_angles=variant_angles, n_pixels=n_idx)
    dataset._component_angles = result
    return result


def _fractions_from_angles(angles: ComponentAngles, selected: Optional[np.ndarray],
                           tolerance_deg: float) -> Dict[str, Optional[float]]:
    """Percent of the (selected) indexed pixels within the tolerance of each component.

    As the script counts (``volEBSD_X = volume(oriR, sym_X(1), tol)*100 + volume(oriR, sym_X(2), tol)*100 + ...``):
    per component the SUM over its distinct variants of the share of pixels within the tolerance of that variant.
    A pixel within the tolerance of two variants therefore counts twice (the sum can exceed the share of pixels
    that are near the component). ``selected``: bool over the pixels of the phase (None: all of them); the share is
    taken of the selected pixels. A component that could not be built, or a selection without pixels, gives None (a
    missing number, an empty cell in the workbook, never a 0 that reads like a measurement);
    a NaN orientation is in the denominator and in no count.
    """
    tolerance_rad = np.radians(tolerance_deg)
    n_ori = int(angles.n_pixels if selected is None else np.count_nonzero(selected))
    out: Dict[str, float] = {}
    for k, name in enumerate(angles.names):
        if n_ori == 0 or angles.failed[k]:
            out[name] = None
            continue
        block = angles.variant_angles[k] if selected is None else angles.variant_angles[k][:, selected]
        # one volume(...)*100 per variant, added in the order of the variants, as the script adds them
        out[name] = sum((np.sum(row < tolerance_rad) / n_ori) * 100.0 for row in block)
    return out


def calculate_texture_component_fractions(dataset, components: Dict[str, "TextureComponent"],
                                         tolerance_deg: float = 10.0,
                                         pixel_mask: Optional[np.ndarray] = None,
                                         texture_phase=MOST_MEASURED) -> Dict[str, Optional[float]]:
    """Calculate volume fractions for all texture components in a preset.

    Counted as ``TextureAnalysis_mtex6.m`` lines 154-167 count them: per component the sum over its distinct
    variants (``component_variants``) of the share of pixels within the tolerance of that variant, so a pixel
    within the tolerance of two variants of one component counts twice. The share is of all pixels of the texture phase
    (the phase with the most indexed pixels, or ``texture_phase``; NaN orientations included in the count of pixels, in
    no variant's count); pixels of other phases and not-indexed pixels are left out, as ``ebsd(ph)`` leaves them out.

    Args:
        dataset: EBSDDataset instance
        components: Dictionary of {component_name: TextureComponent} from TEXTURE_PRESETS
        tolerance_deg: Tolerance angle for volume fraction calculation (degrees)
        pixel_mask: Optional boolean mask over all pixels (shape (size,) or (ny, nx)); the fractions are then
            the share of the phase's pixels inside the mask (a subset of the grains, e.g. the RX grains).
            None: all pixels of the phase.
        texture_phase: "most_measured" (default) or a phase id (``resolve_texture_phase``)

    Returns:
        Dictionary of {component_name: volume_fraction_percent}; None for a component that could not be built

    Reference: TextureAnalysis_mtex6.m lines 153-207
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    pid = _phase_id_of(dataset, texture_phase)
    phase_mask = np.asarray(dataset.xmap.phase_id) == pid
    selected = None
    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool).reshape(-1)
        if pixel_mask.size != phase_mask.size:
            raise ValueError(f"pixel_mask has {pixel_mask.size} pixels, the dataset {phase_mask.size}")
        selected = pixel_mask[phase_mask]                   # over the pixels of the phase, in map order

    # the angle of every pixel of the phase to every component: computed once per dataset
    return _fractions_from_angles(get_component_angles(dataset, components, pid), selected, tolerance_deg)


def split_phase_conflict(rx_result, texture_result) -> Optional[str]:
    """Why the RX / substructure split cannot be made, or None when it can.

    The split counts the pixels of the texture phase in the RX grains and in the other grains. The RX grains are
    grains of the RX phase, so when that is not the texture phase no RX grain holds a pixel of the texture phase and the
    "substructure" group would be every grain of the texture phase, which no RX classification was made on. The
    callers leave the split (the sheet "SmallVSLargeGrain") out and say this sentence in their warnings.
    """
    rx_id, tx_id = getattr(rx_result, "phase_id", None), getattr(texture_result, "phase_id", None)
    if rx_id is None or tx_id is None or rx_id == tx_id:
        return None
    return (f"the RX phase ({getattr(rx_result, 'phase_name', None) or rx_id}) is not the texture phase "
            f"({getattr(texture_result, 'phase_name', None) or tx_id}); the split needs one phase, so the sheet is left out")


def calculate_texture_split_by_rx(dataset, grains, rx_grain_ids, components: Dict[str, "TextureComponent"],
                                  tolerance_deg: float = 10.0, texture_phase=MOST_MEASURED) -> Dict[str, Any]:
    """Texture-component volume fractions of the RX grains and of the other (substructure) grains.

    The sheet "SmallVSLargeGrain" of Documentation.xlsx puts these next to the fractions of all grains
    (TextureAnalysis_mtex6.m "Seperate by GOS", lines 648-720). The split is the one
    ``analyze_recrystallization`` made (``RXResult.rx_grain_ids``); the components are those of the
    sheet-1 fractions, so pass the same ``components``, ``tolerance_deg`` and ``texture_phase`` as for
    ``analyze_texture``. Only the pixels of the texture phase are counted, in the pixel numbers too.

    Returns:
        {"rx": {component: percent or None}, "substructure": {...},
         "n_rx_pixels": int, "n_substructure_pixels": int}
        A group without indexed pixels has None for every component (not 0 %).
        The exporter takes this dict as ``results["texture_by_rx"]``.
    """
    grain_ids = np.asarray(grains.grain_ids).reshape(-1)
    rx_mask = np.isin(grain_ids, np.asarray(rx_grain_ids, dtype=int)) & (grain_ids > 0)
    sub_mask = (grain_ids > 0) & ~rx_mask
    pid = _phase_id_of(dataset, texture_phase)
    indexed = np.asarray(dataset.xmap.phase_id) == pid
    # the angles to the components do not depend on the group: the fractions of the whole map computed them
    # (or this call does, once), the two groups only pick their pixels
    angles = get_component_angles(dataset, components, pid)
    out: Dict[str, Any] = {}
    for key, mask in (("rx", rx_mask), ("substructure", sub_mask)):
        n_px = int(np.sum(mask & indexed))
        out["n_rx_pixels" if key == "rx" else "n_substructure_pixels"] = n_px
        if n_px == 0:
            out[key] = {name: None for name in components}
        else:
            out[key] = _fractions_from_angles(angles, mask[indexed], tolerance_deg)
    return out


def _specimen_z_in_crystal_frame(quats: np.ndarray) -> np.ndarray:
    """(n, 3): the specimen Z axis written in the crystal frame, for orix quaternions (w, x, y, z).

    An orix quaternion is the conjugate of the MTEX rotation (crystal -> specimen), so the specimen Z axis seen
    from the crystal is the third column of the rotation matrix of the quaternion as stored. The angle between a
    crystal vector turned into the specimen frame and Z is the angle between that vector and this column.
    NaN quaternions give NaN rows.
    """
    w, x, y, z = np.moveaxis(np.asarray(quats, dtype=np.float64).reshape(-1, 4), -1, 0)
    return np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=-1)


def _phase_quaternions(dataset, phase_id: int) -> np.ndarray:
    """(n, 4) orix quaternions as stored, of the pixels of a phase (no symmetry reduction)."""
    mask = np.asarray(dataset.xmap.phase_id) == int(phase_id)
    return np.asarray(dataset.xmap.rotations[mask].data, dtype=np.float64).reshape(-1, 4)


def calculate_surface_plane_fraction(dataset, miller_hkl: list,
                                     tolerance_deg: float = 15.0, texture_phase=MOST_MEASURED) -> Optional[float]:
    """Percent of pixels whose stored orientation turns the plane normal (h k l) to within the tolerance of +Z.

    This is the script's number (``TextureAnalysis_mtex6.m`` 441-523, ``angle(ori * Miller(h,k,l,CS), vector3d.Z)
    < tol``): ONE specimen vector per pixel from the orientation as stored, no symmetry reduction, the angle to +Z
    not antipodal (a vector along -Z is 180 deg away). Aztec stores each orientation in a reduced Euler range where
    (1 0 0) and (1 1 0) hardly ever come near Z, so those two are about 0 % on Aztec data;
    ``calculate_surface_plane_family_fraction`` gives the share with any plane of the family in the surface.

    Only the pixels of the texture phase are counted, as for the components, and they are the denominator; a NaN
    orientation is in the denominator and in no count.

    Args:
        dataset: EBSDDataset instance (provides orientations and phase)
        miller_hkl: Miller indices [h, k, l]
        tolerance_deg: Maximum angle to count as "in the surface" (degrees)
        texture_phase: "most_measured" (default) or a phase id (``resolve_texture_phase``)

    Returns:
        Percent of the phase's pixels (0-100); None when the phase has no pixel

    Reference: TextureAnalysis_mtex6.m lines 441-523
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for surface plane fraction")

    pid = _phase_id_of(dataset, texture_phase)
    n_vec = _unit_miller("hkl", miller_hkl, _phase_object(dataset, pid), "surface plane")
    quats = _phase_quaternions(dataset, pid)
    if len(quats) == 0:
        return None                                  # no pixel, no share: a missing number, not 0 %
    cos_angle = _specimen_z_in_crystal_frame(quats) @ n_vec
    angles = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.count_nonzero(angles < np.radians(tolerance_deg)) / len(quats) * 100.0)


def calculate_surface_plane_family_fraction(dataset, miller_hkl: list,
                                            tolerance_deg: float = 15.0, chunk: int = 65536,
                                            texture_phase=MOST_MEASURED) -> Optional[float]:
    """Percent of pixels that have ANY plane of the family {hkl} within the tolerance of the surface normal Z.

    This is what the column title "pct of 111 plane in surface" means physically: all crystal equivalents of the
    plane normal under the phase's proper point group are tried, a normal and its negative are the same plane (the
    angle is that between the plane normal and the Z line, at most 90 deg), and a pixel counts when the smallest
    angle is below the tolerance. Unlike ``calculate_surface_plane_fraction`` it does not depend on which symmetric
    equivalent the file stores. Same pixels and denominator as there (the texture phase).

    Args:
        dataset: EBSDDataset instance
        miller_hkl: Miller indices [h, k, l] of the plane family
        tolerance_deg: Maximum angle between a plane normal of the family and Z (degrees)
        chunk: pixels per block of the (pixels x equivalents) work array
        texture_phase: "most_measured" (default) or a phase id (``resolve_texture_phase``)

    Returns:
        Percent of the phase's pixels (0-100); None when the phase has no pixel
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for surface plane fraction")
    from analysis.grain_reconstruction import phase_ops

    pid = _phase_id_of(dataset, texture_phase)
    ops = phase_ops(dataset.xmap).get(pid)
    if ops is None:
        raise ValueError(f"phase {pid} has no point group; the plane family is not defined")
    n_vec = _unit_miller("hkl", miller_hkl, _phase_object(dataset, pid), "surface plane")
    family = _matrices(ops) @ n_vec                                  # (k, 3): the equivalents of the normal
    quats = _phase_quaternions(dataset, pid)
    if len(quats) == 0:
        return None                                  # no pixel, no share: a missing number, not 0 %
    z_crystal = _specimen_z_in_crystal_frame(quats)
    tolerance_rad = np.radians(tolerance_deg)
    hits = 0
    for i in range(0, len(quats), chunk):
        best = np.abs(z_crystal[i:i + chunk] @ family.T).max(axis=1)    # NaN stays NaN and never counts
        hits += int(np.count_nonzero(np.arccos(np.clip(best, 0.0, 1.0)) < tolerance_rad))
    return float(hits / len(quats) * 100.0)


def calculate_texture_component_map(dataset, components: Dict[str, "TextureComponent"],
                                    tolerance_deg: float = 10.0, texture_phase=MOST_MEASURED) -> Tuple[np.ndarray, List[str]]:
    """Assign each pixel to its best-matching texture component.

    Returns a 2D integer map where each pixel value is:
    - 0 = no component matched within tolerance (also a pixel with a NaN orientation)
    - 1..N = 1-indexed component ID (order matches components dict); the nearest component, the lowest index
      when two are within 1e-9 rad of each other
    Pixels of other phases and not-indexed pixels are 0: only the pixels of the texture phase are classified.

    Parameters
    ----------
    dataset : EBSDDataset
        Must provide orientations_2d, indexed_mask_2d, phase.
    components : dict
        {name: TextureComponent} from TEXTURE_PRESETS.
    tolerance_deg : float
        Maximum angular distance to assign a component.
    texture_phase : "most_measured" or a phase id
        The phase whose pixels are classified (``resolve_texture_phase``).

    Returns
    -------
    component_map : ndarray of int, shape (ny, nx)
        Component assignment per pixel (0 = unmatched).
    component_names : list of str
        Component names in order (index 1 = first name, etc.).
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    ny, nx = dataset.shape
    pid = _phase_id_of(dataset, texture_phase)
    indexed = (np.asarray(dataset.xmap.phase_id) == pid).reshape(ny, nx)
    n_indexed = int(indexed.sum())

    if n_indexed == 0:
        return np.zeros((ny, nx), dtype=np.int32), list(components.keys())

    tolerance_rad = np.radians(tolerance_deg)
    # Track best component per indexed pixel
    best_angle = np.full(n_indexed, np.inf)
    best_comp = np.zeros(n_indexed, dtype=np.int32)  # 0 = no match

    # the angle of every indexed pixel to every component: computed once per dataset
    angles = get_component_angles(dataset, components, pid)

    for comp_idx in range(1, len(angles.names) + 1):
        min_angles = angles.component_min_angle(comp_idx - 1)
        # (a component that could not be built has NaN angles and claims no pixel; nor does a NaN orientation)
        # a later component takes a pixel only if it is closer by more than MAP_TIE_MARGIN: a tie goes to the
        # lowest component index
        improved = (min_angles < best_angle - MAP_TIE_MARGIN) & (min_angles < tolerance_rad)
        best_angle[improved] = min_angles[improved]
        best_comp[improved] = comp_idx

    # Map back to 2D
    component_map = np.zeros((ny, nx), dtype=np.int32)
    component_map[indexed] = best_comp

    return component_map, list(components.keys())


def analyze_texture(dataset, components: Dict[str, "TextureComponent"],
                   tolerance_deg: float = 10.0,
                   surface_plane_tolerance_deg: float = 15.0,
                   texture_phase=MOST_MEASURED,
                   preset_name: Optional[str] = None) -> TextureResult:
    """Full texture analysis workflow.

    Everything is computed on ONE phase, the texture phase: the phase with the most indexed pixels of the map
    (default) or the phase id ``texture_phase`` names (``resolve_texture_phase``; it must be cubic). The components
    are built for it and only its pixels are counted, in the component fractions, the surface plane fractions
    and the texture index and entropy; the result names the phase (``phase_id``, ``phase_name``).

    Texture index and entropy are the script's ``norm(odf)^2`` and ``entropy(odf)`` (TextureAnalysis_mtex6.m lines
    101-102) of the 5 degree de la Vallee Poussin kernel density of the texture phase's orientations
    (``analysis.texture_odf.texture_index_and_entropy``): a uniform texture has index 1 and entropy 0. They are None
    when the phase has no pixel with a finite orientation.

    Args:
        dataset: EBSDDataset instance
        components: Dictionary of texture components (from TEXTURE_PRESETS)
        tolerance_deg: Tolerance angle for component volume fractions (degrees)
        surface_plane_tolerance_deg: Tolerance for surface plane fractions (degrees)
        texture_phase: "most_measured" (default) or a phase id
        preset_name: the name of the preset ``components`` came from (recorded in the result and the workbook)

    Returns:
        TextureResult with all metrics

    Raises:
        ValueError: ``texture_phase`` is not a phase of the map, or the phase is not cubic

    Reference: TextureAnalysis_mtex6.m full workflow
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    pid = resolve_texture_phase(dataset, texture_phase)

    # 1. Calculate texture component volume fractions
    component_fractions = calculate_texture_component_fractions(
        dataset, components, tolerance_deg, texture_phase=pid
    )

    # 2. Calculate surface plane fractions
    plane_111_frac = calculate_surface_plane_fraction(
        dataset, [1, 1, 1], surface_plane_tolerance_deg, texture_phase=pid
    )
    plane_100_frac = calculate_surface_plane_fraction(
        dataset, [1, 0, 0], surface_plane_tolerance_deg, texture_phase=pid
    )
    plane_110_frac = calculate_surface_plane_fraction(
        dataset, [1, 1, 0], surface_plane_tolerance_deg, texture_phase=pid
    )
    family_fracs = [calculate_surface_plane_family_fraction(dataset, hkl, surface_plane_tolerance_deg,
                                                            texture_phase=pid)
                    for hkl in ([1, 1, 1], [1, 0, 0], [1, 1, 0])]

    # 3. Texture index and entropy of the kernel density, on the same pixels as the component fractions
    from analysis.texture_odf import texture_index_and_entropy
    phase_mask = np.asarray(dataset.xmap.phase_id) == pid
    finite = np.isfinite(np.asarray(dataset.xmap.rotations.data, dtype=float).reshape(-1, 4)[phase_mask]).all(axis=1)
    if finite.any():
        texture_index, entropy = texture_index_and_entropy(dataset, phase_mask, halfwidth_deg=5.0)
    else:
        # the phase has no pixel with a finite orientation: there is no density to measure
        texture_index = entropy = None

    return TextureResult(
        component_fractions=component_fractions,
        plane_111_fraction=plane_111_frac,
        plane_100_fraction=plane_100_frac,
        plane_110_fraction=plane_110_frac,
        texture_index=texture_index,
        entropy=entropy,
        specimen_rotation=None,  # No auto-centering for now
        tolerance_angle=tolerance_deg,
        plane_111_family_fraction=family_fracs[0],
        plane_100_family_fraction=family_fracs[1],
        plane_110_family_fraction=family_fracs[2],
        phase_id=pid,
        phase_name=_phase_label(dataset, pid),
        preset_name=preset_name,
    )
