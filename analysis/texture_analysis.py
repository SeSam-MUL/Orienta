"""
Texture Analysis Module for EBSD Data.

Implements:
- Volume fraction calculation for texture components (14 FCC rolling components)
- Surface plane fraction calculation for {111}, {100}, {110}
- ODF estimation (simplified kernel density approach)
- Texture index and entropy calculation

Reference: MATLAB/MTEX TextureAnalysis_mtex6.m lines 149-519
Architecture: the MTEX-equivalence design notes §2.4
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

try:
    from orix.quaternion import Misorientation, Orientation
    from orix.vector import Miller, Vector3d
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False


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

    Reference: TextureAnalysis_mtex6.m lines 150-163 (EBSD-based volume fractions)
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
                                         tolerance_deg: float = 10.0) -> Dict[str, float]:
    """Calculate volume fractions for all texture components in a preset.

    Args:
        dataset: EBSDDataset instance
        components: Dictionary of {component_name: TextureComponent} from TEXTURE_PRESETS
        tolerance_deg: Tolerance angle for volume fraction calculation (degrees)

    Returns:
        Dictionary of {component_name: volume_fraction_percent}

    Reference: TextureAnalysis_mtex6.m lines 149-203
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for texture analysis")

    # Get phase and orientations
    phase = dataset.phase
    orientations = dataset.orientations  # Orientation array with symmetry

    # Get indexed orientations only
    indexed_mask = dataset.indexed_mask
    ori_flat = orientations[indexed_mask]

    # Specimen symmetry (orthorhombic mmm for rolling)
    from orix.quaternion import symmetry
    specimen_symmetry = symmetry.D2h  # mmm

    results = {}

    for comp_name, comp in components.items():
        # Create ideal orientation from Miller indices or Euler angles
        try:
            if comp.hkl is not None and comp.uvw is not None:
                miller_hkl = Miller(hkl=comp.hkl, phase=phase)
                miller_uvw = Miller(uvw=comp.uvw, phase=phase)
                ideal_ori = Orientation.from_align_vectors(
                    miller_uvw, miller_hkl
                )
            elif comp.euler_deg is not None:
                euler_rad = np.radians(comp.euler_deg)
                ideal_ori = Orientation.from_euler(
                    euler_rad, symmetry=phase.point_group
                )
            else:
                raise ValueError(f"{comp_name}: No valid definition (Miller or Euler)")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Skipping texture component %s: %s", comp_name, e)
            results[comp_name] = 0.0
            continue

        # Apply specimen symmetry to get all equivalents
        ideal_with_sym = specimen_symmetry.outer(ideal_ori).reshape(-1).unique()

        # Calculate volume fraction (with crystal symmetry for correct angles)
        vol_frac = calculate_volume_fraction(
            ori_flat, ideal_with_sym, tolerance_deg,
            symmetry=phase.point_group
        )
        results[comp_name] = vol_frac

    return results


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

    Reference: TextureAnalysis_mtex6.m lines 437-519

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
    indexed_mask = dataset.indexed_mask
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

    from orix.quaternion import symmetry as orix_sym

    phase = dataset.phase
    crystal_sym = phase.point_group
    specimen_symmetry = orix_sym.D2h

    ori_2d = dataset.orientations_2d
    indexed = dataset.indexed_mask_2d
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
            if comp.hkl is not None and comp.uvw is not None:
                miller_hkl = Miller(hkl=comp.hkl, phase=phase)
                miller_uvw = Miller(uvw=comp.uvw, phase=phase)
                ideal_ori = Orientation.from_align_vectors(miller_uvw, miller_hkl)
            elif comp.euler_deg is not None:
                euler_rad = np.radians(comp.euler_deg)
                ideal_ori = Orientation.from_euler(euler_rad, symmetry=crystal_sym)
            else:
                continue
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Skipping texture component %s in map: %s", comp_name, e)
            continue

        ideal_with_sym = specimen_symmetry.outer(ideal_ori).reshape(-1).unique()

        # Find min misorientation angle for each pixel to any symmetry equivalent
        min_angles = np.full(n_indexed, np.inf)
        sym_pair = (crystal_sym, crystal_sym)
        for ideal in ideal_with_sym:
            miso = Misorientation(ori_flat * (~ideal), symmetry=sym_pair)
            miso = miso.reduce()  # symmetry-reduced fundamental zone (orix>=0.15 name)
            min_angles = np.minimum(min_angles, miso.angle)

        # Update best assignment where this component is closer
        improved = (min_angles < best_angle) & (min_angles < tolerance_rad)
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
    fracs = np.array(list(component_fractions.values())) / 100.0  # % → fraction
    total_assigned = fracs.sum()
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
    fracs = np.array(list(component_fractions.values())) / 100.0
    total_assigned = fracs.sum()
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
