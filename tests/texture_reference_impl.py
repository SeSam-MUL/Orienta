"""The texture code as it was before the per-pixel component angles were computed once (analysis/texture_analysis.py at
commit 1329d9fb), kept as the independent reference for tests/test_texture_angles_reuse.py: every fraction, the split and
the component map of the new code must equal what these functions return, bit for bit. Do not import this from the app.

This is the reference for the ANGLE ARITHMETIC (orix ``Misorientation(...).reduce()``) and for the structure of the
computation only. How a component's variants are built is NOT here any more: the construction used to be wrong (a
single-pair ``from_align_vectors``, so Inv. Goss equalled Cube, and the specimen symmetry collapsed) and is tested against
MTEX in tests/test_texture_components_mtex.py. The tests hand the variants in with ``set_variant_builder``; the
reference then counts as the script of the owner counts: per component the SUM over its variants of the share of
pixels within the tolerance, and a component map in which the lowest index keeps a tie within 1e-9 rad."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

try:
    from orix.quaternion import Misorientation, Orientation
    from orix.vector import Miller, Vector3d
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

_VARIANTS = {"builder": None}
MAP_TIE_MARGIN = 1e-9


def _phase_mask(dataset):
    """The pixels of the components' phase: the first indexed phase of the map (the script's ``ebsd(ph)``)."""
    ids = [int(i) for i in dataset.xmap.phases.ids if int(i) >= 0]
    return np.asarray(dataset.xmap.phase_id) == ids[0]


def set_variant_builder(fn):
    """``fn(component, phase) -> orix Orientation`` (n variants); the tests supply the construction under test."""
    _VARIANTS["builder"] = fn


def variants_of(component, phase):
    if _VARIANTS["builder"] is None:
        raise RuntimeError("tests.texture_reference_impl needs set_variant_builder(...) first")
    return _VARIANTS["builder"](component, phase)


@dataclass
class TextureResult:
    """Results from texture analysis.

    Attributes:
        # Texture component volume fractions (14 components for FCC rolling)
        component_fractions: Dict[str, float]  # Component name → volume fraction (%)

        # Surface plane fractions
        plane_111_fraction: float  # Fraction with {111} plane in surface (%)
        plane_100_fraction: float  # Fraction with {100} plane in surface (%)
        plane_110_fraction: float  # Fraction with {110} plane in surface (%)

        # ODF-based metrics
        texture_index: Optional[float]  # Texture index (L2 norm squared)
        entropy: Optional[float]  # Texture entropy

        # Preprocessing
        specimen_rotation: Optional[np.ndarray]  # Applied specimen rotation (Euler angles, degrees)
        tolerance_angle: float  # Tolerance angle used for volume fractions (degrees)
    """
    component_fractions: Dict[str, float]
    plane_111_fraction: float
    plane_100_fraction: float
    plane_110_fraction: float
    texture_index: Optional[float] = None
    entropy: Optional[float] = None
    specimen_rotation: Optional[np.ndarray] = None
    tolerance_angle: float = 10.0


def calculate_volume_fraction(orientations: "Orientation", ideal_orientations: "Orientation",
                              tolerance_deg: float = 10.0,
                              symmetry=None) -> float:
    """Calculate volume fraction of orientations matching ideal component.

    For each EBSD orientation, finds the minimum misorientation angle to any of the
    ideal orientations (symmetry equivalents). Counts orientations within tolerance.

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

    # For each EBSD orientation, calculate misorientation to ALL ideal orientations
    # We need to find the minimum angle across all symmetry equivalents

    n_ori = orientations.size
    if n_ori == 0:
        return 0.0

    min_angles = np.full(n_ori, np.inf)

    sym_pair = (symmetry, symmetry) if symmetry is not None else None

    for ideal_ori in ideal_orientations:
        # Misorientation from all EBSD orientations to this ideal orientation
        # Must include crystal symmetry for correct minimum angle calculation
        miso = Misorientation(
            orientations * (~ideal_ori),
            symmetry=sym_pair
        ) if sym_pair else Misorientation(orientations * (~ideal_ori))
        # Attaching symmetry to the constructor is NOT enough — orix returns
        # the raw angle until the misorientation is mapped into the
        # symmetry-reduced fundamental zone. Without this, crystal-symmetry
        # equivalents within tolerance are missed and every volume fraction
        # (and the texture index/entropy derived from them) is undercounted.
        # Mirror the working sibling in calculate_texture_component_map.
        if sym_pair is not None:
            miso = miso.reduce()  # symmetry-reduced fundamental zone (orix>=0.15 name)
        angles = miso.angle  # shape (n_ori,)

        # Update minimum
        min_angles = np.minimum(min_angles, angles)

    # Count orientations within tolerance
    within_tolerance = np.sum(min_angles < tolerance_rad)
    volume_fraction = (within_tolerance / n_ori) * 100.0

    return volume_fraction


def calculate_texture_component_fractions(dataset, components: Dict[str, "TextureComponent"],
                                         tolerance_deg: float = 10.0,
                                         pixel_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate volume fractions for all texture components in a preset.

    Args:
        dataset: EBSDDataset instance
        components: Dictionary of {component_name: TextureComponent} from TEXTURE_PRESETS
        tolerance_deg: Tolerance angle for volume fraction calculation (degrees)
        pixel_mask: Optional boolean mask over all pixels (shape (size,) or (ny, nx)); the fractions are then
            the share of the indexed pixels inside the mask (a subset of the grains, e.g. the RX grains).
            None: all indexed pixels, as before.

    Returns:
        Dictionary of {component_name: volume_fraction_percent}

    Reference: TextureAnalysis_mtex6.m lines 153-207
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    # Get phase and orientations
    phase = dataset.phase
    orientations = dataset.orientations  # Orientation array with symmetry

    # Get indexed orientations only
    indexed_mask = _phase_mask(dataset)
    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool).reshape(-1)
        if pixel_mask.size != indexed_mask.size:
            raise ValueError(f"pixel_mask has {pixel_mask.size} pixels, the dataset {indexed_mask.size}")
        indexed_mask = indexed_mask & pixel_mask
    ori_flat = orientations[indexed_mask]

    results = {}

    for comp_name, comp in components.items():
        try:
            variants = variants_of(comp, phase)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Skipping texture component %s: %s", comp_name, e)
            results[comp_name] = None        # the one change since: a missing number, no longer a 0.0
            continue

        # the script's sum over the variants of volume(oriR, variant, tol) (crystal symmetry for the angles)
        results[comp_name] = sum(
            calculate_volume_fraction(ori_flat, variants[j:j + 1], tolerance_deg, symmetry=phase.point_group)
            for j in range(variants.size))

    return results


def calculate_texture_split_by_rx(dataset, grains, rx_grain_ids, components: Dict[str, "TextureComponent"],
                                  tolerance_deg: float = 10.0) -> Dict[str, Any]:
    """Texture-component volume fractions of the RX grains and of the other (substructure) grains.

    The sheet "SmallVSLargeGrain" of Documentation.xlsx puts these next to the fractions of all grains
    (TextureAnalysis_mtex6.m "Seperate by GOS", lines 648-720). The split is the one
    ``analyze_recrystallization`` made (``RXResult.rx_grain_ids``); the components are those of the
    sheet-1 fractions, so pass the same ``components`` and ``tolerance_deg`` as for ``analyze_texture``.

    Returns:
        {"rx": {component: percent or None}, "substructure": {...},
         "n_rx_pixels": int, "n_substructure_pixels": int}
        A group without indexed pixels has None for every component (not 0 %).
        The exporter takes this dict as ``results["texture_by_rx"]``.
    """
    grain_ids = np.asarray(grains.grain_ids).reshape(-1)
    rx_mask = np.isin(grain_ids, np.asarray(rx_grain_ids, dtype=int)) & (grain_ids > 0)
    sub_mask = (grain_ids > 0) & ~rx_mask
    indexed = _phase_mask(dataset)
    out: Dict[str, Any] = {}
    for key, mask in (("rx", rx_mask), ("substructure", sub_mask)):
        n_px = int(np.sum(mask & indexed))
        out["n_rx_pixels" if key == "rx" else "n_substructure_pixels"] = n_px
        if n_px == 0:
            out[key] = {name: None for name in components}
        else:
            out[key] = calculate_texture_component_fractions(dataset, components, tolerance_deg,
                                                             pixel_mask=mask)
    return out


def calculate_surface_plane_fraction(dataset, miller_hkl: list,
                                     tolerance_deg: float = 15.0) -> float:
    """Calculate fraction of orientations with given plane in surface plane.

    The surface plane is assumed to be perpendicular to the sample Z-direction.
    Calculates the angle between the Miller plane normal (transformed by orientation)
    and the Z-direction. Counts orientations where this angle is within tolerance.

    Args:
        dataset: EBSDDataset instance (provides orientations and phase)
        miller_hkl: Miller indices [h, k, l] for the plane
        tolerance_deg: Maximum angle to count as "in surface plane" (degrees)

    Returns:
        Volume fraction in percent (0-100)

    Reference: TextureAnalysis_mtex6.m lines 441-523

    MATLAB logic:
    ```matlab
    miller111 = Miller(1,1,1, ebsd.orientations.CS);
    zDirection = vector3d.Z;
    angleToZ = angle(ebsd.orientations * miller111, zDirection);
    alignedOrientations = angleToZ < tolerance;
    volumeFraction = sum(alignedOrientations) / numel(alignedOrientations) * 100;
    ```
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for surface plane fraction")

    tolerance_rad = np.radians(tolerance_deg)

    # Get indexed orientations only
    indexed_mask = _phase_mask(dataset)
    orientations = dataset.orientations[indexed_mask]

    # Get phase from dataset
    phase = dataset.phase

    # Miller vector representing plane normal
    miller_vec = Miller(hkl=miller_hkl, phase=phase)

    # Z-direction in sample coordinate system
    z_direction = Vector3d.zvector()

    # Rotate Miller plane normal by each orientation
    # This gives the crystal direction in the sample frame
    rotated_normals = orientations * miller_vec

    # Convert Miller vectors to Vector3d for angle calculation
    # Miller vectors are already in Cartesian coordinates after rotation
    rotated_vec3d = Vector3d(rotated_normals.data)

    # Calculate angle to Z-direction
    angles_to_z = rotated_vec3d.angle_with(z_direction)

    # Count orientations within tolerance
    aligned = np.sum(angles_to_z < tolerance_rad)
    volume_fraction = (aligned / orientations.size) * 100.0

    return volume_fraction


def calculate_texture_component_map(dataset, components: Dict[str, "TextureComponent"],
                                    tolerance_deg: float = 10.0) -> np.ndarray:
    """Assign each pixel to its best-matching texture component.

    Returns a 2D integer map where each pixel value is:
    - 0 = no component matched within tolerance
    - 1..N = 1-indexed component ID (order matches components dict)

    Parameters
    ----------
    dataset : EBSDDataset
        Must provide orientations_2d, indexed_mask_2d, phase.
    components : dict
        {name: TextureComponent} from TEXTURE_PRESETS.
    tolerance_deg : float
        Maximum angular distance to assign a component.

    Returns
    -------
    component_map : ndarray of int, shape (ny, nx)
        Component assignment per pixel (0 = unmatched).
    component_names : list of str
        Component names in order (index 1 = first name, etc.).
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    phase = dataset.phase
    crystal_sym = phase.point_group

    ori_2d = dataset.orientations_2d
    indexed = _phase_mask(dataset).reshape(dataset.shape)
    ny, nx = ori_2d.shape

    # Flatten indexed orientations
    ori_flat = ori_2d[indexed]
    n_indexed = ori_flat.size

    if n_indexed == 0:
        return np.zeros((ny, nx), dtype=np.int32), list(components.keys())

    tolerance_rad = np.radians(tolerance_deg)
    # Track best component per indexed pixel
    best_angle = np.full(n_indexed, np.inf)
    best_comp = np.zeros(n_indexed, dtype=np.int32)  # 0 = no match

    for comp_idx, (comp_name, comp) in enumerate(components.items(), 1):
        try:
            ideal_with_sym = variants_of(comp, phase)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Skipping texture component %s in map: %s", comp_name, e)
            continue

        # Find min misorientation angle for each pixel to any symmetry equivalent
        min_angles = np.full(n_indexed, np.inf)
        sym_pair = (crystal_sym, crystal_sym)
        for ideal in ideal_with_sym:
            miso = Misorientation(ori_flat * (~ideal), symmetry=sym_pair)
            miso = miso.reduce()  # symmetry-reduced fundamental zone (orix>=0.15 name)
            min_angles = np.minimum(min_angles, miso.angle)

        # Update best assignment where this component is closer
        improved = (min_angles < best_angle - MAP_TIE_MARGIN) & (min_angles < tolerance_rad)
        best_angle[improved] = min_angles[improved]
        best_comp[improved] = comp_idx

    # Map back to 2D
    component_map = np.zeros((ny, nx), dtype=np.int32)
    component_map[indexed] = best_comp

    return component_map, list(components.keys())


def _compute_texture_index(component_fractions: Dict[str, float]) -> float:
    """Compute texture index (J-index) from component volume fractions.

    The texture index measures texture sharpness. J = sum(fi²) where fi are
    fractional volumes (0-1). For a single-component texture J→1, for random
    texture J→0.

    Uses component fractions plus the "unassigned" fraction as bins
    approximating the ODF.

    Reference: MTEX TextureAnalysis_mtex6.m (textureindex calculation)
    """
    fracs = np.array([0.0 if v is None else v for v in component_fractions.values()]) / 100.0  # % → fraction
    total_assigned = fracs.sum()
    if total_assigned > 1.0:            # summed (double-counted) fractions: divided by their sum
        fracs = fracs / total_assigned
        total_assigned = 1.0
    remainder = max(0.0, 1.0 - total_assigned)
    all_fracs = np.append(fracs, remainder)
    # Filter out zero bins
    nonzero = all_fracs[all_fracs > 0]
    if len(nonzero) == 0:
        return 1.0
    return float(np.sum(nonzero ** 2))


def _compute_texture_entropy(component_fractions: Dict[str, float]) -> float:
    """Compute texture entropy from component volume fractions.

    S = -sum(fi * ln(fi)) where fi > 0. Higher entropy = more random texture.
    Maximum entropy for N bins = ln(N).

    Reference: MTEX TextureAnalysis_mtex6.m (entropy calculation)
    """
    fracs = np.array([0.0 if v is None else v for v in component_fractions.values()]) / 100.0
    total_assigned = fracs.sum()
    if total_assigned > 1.0:
        fracs = fracs / total_assigned
        total_assigned = 1.0
    remainder = max(0.0, 1.0 - total_assigned)
    all_fracs = np.append(fracs, remainder)
    nonzero = all_fracs[all_fracs > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)))


def analyze_texture(dataset, components: Dict[str, "TextureComponent"],
                   tolerance_deg: float = 10.0,
                   surface_plane_tolerance_deg: float = 15.0) -> TextureResult:
    """Full texture analysis workflow.

    Args:
        dataset: EBSDDataset instance
        components: Dictionary of texture components (from TEXTURE_PRESETS)
        tolerance_deg: Tolerance angle for component volume fractions (degrees)
        surface_plane_tolerance_deg: Tolerance for surface plane fractions (degrees)

    Returns:
        TextureResult with all metrics

    Reference: TextureAnalysis_mtex6.m full workflow
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    # 1. Calculate texture component volume fractions
    component_fractions = calculate_texture_component_fractions(
        dataset, components, tolerance_deg
    )

    # 2. Calculate surface plane fractions
    plane_111_frac = calculate_surface_plane_fraction(
        dataset, [1, 1, 1], surface_plane_tolerance_deg
    )
    plane_100_frac = calculate_surface_plane_fraction(
        dataset, [1, 0, 0], surface_plane_tolerance_deg
    )
    plane_110_frac = calculate_surface_plane_fraction(
        dataset, [1, 1, 0], surface_plane_tolerance_deg
    )

    # 3. Texture metrics from component volume fractions
    texture_index = _compute_texture_index(component_fractions)
    entropy = _compute_texture_entropy(component_fractions)

    return TextureResult(
        component_fractions=component_fractions,
        plane_111_fraction=plane_111_frac,
        plane_100_fraction=plane_100_frac,
        plane_110_fraction=plane_110_frac,
        texture_index=texture_index,
        entropy=entropy,
        specimen_rotation=None,  # No auto-centering for now
        tolerance_angle=tolerance_deg,
    )


# ---------------------------------------------------------------------------------------------
# for the hand computation in the tests: the per-pixel result of the whole map, which a test then masks itself
# ---------------------------------------------------------------------------------------------
def per_pixel_in_tolerance(dataset, components: Dict[str, "TextureComponent"], tolerance_deg: float = 10.0):
    """int (n_components, n_indexed): for the indexed pixel (flat map order), the NUMBER of the component's variants it
    is within the tolerance of (the script counts a pixel once per variant).

    The loop body of ``calculate_texture_component_fractions`` and ``calculate_volume_fraction``, with the
    per-pixel answer kept instead of its mean. A component that cannot be built is 0 everywhere.
    """
    phase = dataset.phase
    ori_flat = dataset.orientations[_phase_mask(dataset)]
    out = np.zeros((len(components), ori_flat.size), dtype=int)
    for k, (comp_name, comp) in enumerate(components.items()):
        try:
            variants = variants_of(comp, phase)
        except Exception:
            continue
        for j in range(variants.size):
            miso = Misorientation(ori_flat * (~variants[j:j + 1]),
                                  symmetry=(phase.point_group, phase.point_group)).reduce()
            out[k] += miso.angle < np.radians(tolerance_deg)
    return out


def split_by_hand(dataset, grains, rx_grain_ids, components, tolerance_deg: float = 10.0):
    """{"rx": {component: percent or None}, "substructure": {...}}: the whole-map per-pixel answer masked by hand."""
    in_tol = per_pixel_in_tolerance(dataset, components, tolerance_deg)
    ids = np.asarray(grains.grain_ids).reshape(-1)[np.asarray(_phase_mask(dataset))]
    rx = np.isin(ids, np.asarray(rx_grain_ids, dtype=int)) & (ids > 0)
    sub = (ids > 0) & ~rx
    out = {}
    for key, sel in (("rx", rx), ("substructure", sub)):
        out[key] = {name: (None if not sel.any() else 100.0 * in_tol[k, sel].sum() / sel.sum())
                    for k, name in enumerate(components)}
    return out


def min_angle_to_equivalents(orientations, ideal_orientations, symmetry=None) -> np.ndarray:
    """The angle step of the orix implementation (before the vectorised orientation arithmetic): per orientation the
    smallest misorientation angle (radians) to any of the ideal orientations, an orix ``Misorientation`` reduced by the
    crystal symmetry on both sides for every ideal orientation."""
    min_angles = np.full(orientations.size, np.inf)
    sym_pair = (symmetry, symmetry) if symmetry is not None else None
    for ideal_ori in ideal_orientations:
        miso = (Misorientation(orientations * (~ideal_ori), symmetry=sym_pair) if sym_pair
                else Misorientation(orientations * (~ideal_ori)))
        if sym_pair is not None:
            miso = miso.reduce()
        min_angles = np.minimum(min_angles, miso.angle)
    return min_angles


def component_angles(dataset, components: Dict[str, "TextureComponent"], orientations=None):
    """(n_components, n_indexed) angles in radians by the orix implementation; NaN rows for a component that cannot
    be built. ``orientations``: an Orientation array to use instead of the dataset's indexed ones."""
    phase = dataset.phase
    ori_flat = dataset.orientations[_phase_mask(dataset)] if orientations is None else orientations
    out = np.full((len(components), ori_flat.size), np.nan)
    for k, (comp_name, comp) in enumerate(components.items()):
        try:
            variants = variants_of(comp, phase)
        except Exception:
            continue
        out[k] = min_angle_to_equivalents(ori_flat, variants, phase.point_group)
    return out


def variant_angles(dataset, components: Dict[str, "TextureComponent"], orientations=None):
    """List over the components of (n_variants, n_indexed) angles in radians (orix implementation); one NaN row for a
    component that cannot be built."""
    phase = dataset.phase
    ori_flat = dataset.orientations[_phase_mask(dataset)] if orientations is None else orientations
    out = []
    for comp_name, comp in components.items():
        try:
            variants = variants_of(comp, phase)
        except Exception:
            out.append(np.full((1, ori_flat.size), np.nan))
            continue
        out.append(np.stack([min_angle_to_equivalents(ori_flat, variants[j:j + 1], phase.point_group)
                             for j in range(variants.size)]))
    return out


def use_variants_of_the_code_under_test():
    """Hand the variants of ``analysis.texture_analysis.component_variants`` to this reference. The construction itself is
    checked against MTEX in tests/test_texture_components_mtex.py; here only the arithmetic on top of it is compared."""
    from analysis.texture_analysis import component_variants
    set_variant_builder(lambda comp, phase: Orientation(component_variants(comp, phase)))
