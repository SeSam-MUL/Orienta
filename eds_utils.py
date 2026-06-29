"""
EDS Utilities - Extract and convert EDS data from H5OINA files

Provides:
- Raw counts extraction from H5OINA Window Integral datasets
- Simplified Cliff-Lorimer quantification (counts -> Wt.%)
- Wt.% -> At.% conversion using atomic masses
- Phase suggestion based on measured At.% composition

Note: This uses a *simplified* standardless approach. For publication-quality
results, use proper ZAF correction or standards-based quantification.
The purpose here is rapid phase pre-selection for EBSD indexing.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Atomic masses (g/mol) for common EBSD-relevant elements
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906,
    "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87,
    "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60,
    "I": 126.90, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Nd": 144.24,
    "W": 183.84, "Pt": 195.08, "Au": 196.97, "Pb": 207.2, "Bi": 208.98,
    "Th": 232.04, "U": 238.03,
}

# Simplified k-factors relative to Si (SEM-EDS, ~20 kV)
# These are approximate standardless k-factors for common elements.
# Source: Typical values from Oxford/Bruker EDS software defaults.
# For accurate quantification, use measured k-factors from standards.
K_FACTORS_REL_SI: Dict[str, float] = {
    "C": 3.28, "N": 2.56, "O": 1.84, "Na": 0.82, "Mg": 0.78,
    "Al": 0.87, "Si": 1.00, "P": 1.04, "S": 1.01, "K": 1.00,
    "Ca": 1.01, "Ti": 0.93, "V": 0.91, "Cr": 0.90, "Mn": 0.89,
    "Fe": 0.88, "Co": 0.87, "Ni": 0.86, "Cu": 0.85, "Zn": 0.87,
    "Mo": 0.85, "W": 0.75, "Pb": 0.82,
}


@dataclass
class EDSPixelData:
    """EDS data for a single pixel or region."""
    elements: List[str]
    counts: Dict[str, float]         # Raw counts per element
    weight_pct: Dict[str, float]     # Weight percent per element
    atomic_pct: Dict[str, float]     # Atomic percent per element


def parse_element_name(raw_name: str) -> str:
    """
    Extract pure element symbol from H5OINA element names.

    H5OINA can use formats like "Fe Kα1", "Fe Ka1", "Al Ka1", or just "Fe".

    Args:
        raw_name: Raw element name from H5OINA file

    Returns:
        Pure element symbol (e.g., "Fe")
    """
    # Split on space, take first part (handles "Fe Kα1" -> "Fe")
    return raw_name.split()[0].strip()


def counts_to_weight_pct(
    counts: Dict[str, np.ndarray],
    k_factors: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert raw EDS counts to weight percent using simplified Cliff-Lorimer.

    Uses the standardless approach:
        C_i = (I_i / k_i) / sum(I_j / k_j) * 100

    Where I_i are the raw counts and k_i are the k-factors.

    Args:
        counts: Dict mapping element symbol -> counts array (any shape)
        k_factors: Optional custom k-factors. Defaults to K_FACTORS_REL_SI.

    Returns:
        Dict mapping element symbol -> weight percent array (same shape as input)
    """
    if k_factors is None:
        k_factors = K_FACTORS_REL_SI

    if not counts:
        return {}

    # Compute I_i / k_i for each element
    normalized: Dict[str, np.ndarray] = {}
    for elem, data in counts.items():
        k = k_factors.get(elem, 1.0)  # Default k=1.0 for unknown elements
        normalized[elem] = data.astype(np.float64) / k

    # Sum of all normalized intensities (per-pixel)
    total = None
    for arr in normalized.values():
        if total is None:
            total = arr.copy()
        else:
            total += arr

    # Avoid division by zero
    total = np.where(total > 0, total, 1.0)

    # Weight percent = (normalized / total) * 100
    result: Dict[str, np.ndarray] = {}
    for elem, norm_arr in normalized.items():
        result[elem] = (norm_arr / total) * 100.0

    return result


def weight_pct_to_atomic_pct(
    weight_pct: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Convert weight percent to atomic percent.

    Formula: At.%_i = (Wt.%_i / M_i) / sum(Wt.%_j / M_j) * 100

    Where M_i is the atomic mass of element i.

    Args:
        weight_pct: Dict mapping element symbol -> weight percent array

    Returns:
        Dict mapping element symbol -> atomic percent array (same shape)
    """
    if not weight_pct:
        return {}

    # Compute Wt.%_i / M_i for each element
    molar: Dict[str, np.ndarray] = {}
    for elem, wt_arr in weight_pct.items():
        mass = ATOMIC_MASSES.get(elem, 1.0)
        molar[elem] = wt_arr / mass

    # Sum of all molar fractions (per-pixel)
    total = None
    for arr in molar.values():
        if total is None:
            total = arr.copy()
        else:
            total += arr

    # Avoid division by zero
    total = np.where(total > 0, total, 1.0)

    # Atomic percent
    result: Dict[str, np.ndarray] = {}
    for elem, mol_arr in molar.items():
        result[elem] = (mol_arr / total) * 100.0

    return result


def counts_to_all(
    counts: Dict[str, np.ndarray],
    k_factors: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convert raw counts to both weight percent and atomic percent.

    Args:
        counts: Dict mapping element symbol -> counts array
        k_factors: Optional custom k-factors

    Returns:
        Tuple of (counts, weight_pct, atomic_pct) dicts
    """
    wt_pct = counts_to_weight_pct(counts, k_factors)
    at_pct = weight_pct_to_atomic_pct(wt_pct)
    return counts, wt_pct, at_pct


def get_pixel_composition(
    counts: Dict[str, np.ndarray],
    row: int,
    col: int,
    n_cols: int,
    k_factors: Optional[Dict[str, float]] = None,
) -> EDSPixelData:
    """
    Get full EDS composition for a single pixel.

    Args:
        counts: Dict mapping element symbol -> 1D counts array (flattened)
        row: Pixel row index
        col: Pixel column index
        n_cols: Number of columns in the grid
        k_factors: Optional custom k-factors

    Returns:
        EDSPixelData with counts, wt.%, and at.% for the pixel
    """
    idx = row * n_cols + col

    pixel_counts = {}
    for elem, arr in counts.items():
        if idx < len(arr):
            pixel_counts[elem] = float(arr[idx])
        else:
            pixel_counts[elem] = 0.0

    # Convert to arrays for the conversion functions
    pixel_counts_arrays = {
        elem: np.array([val]) for elem, val in pixel_counts.items()
    }

    wt_pct_arrays = counts_to_weight_pct(pixel_counts_arrays, k_factors)
    at_pct_arrays = weight_pct_to_atomic_pct(wt_pct_arrays)

    elements = sorted(pixel_counts.keys())
    wt_pct = {elem: float(wt_pct_arrays[elem][0]) for elem in elements}
    at_pct = {elem: float(at_pct_arrays[elem][0]) for elem in elements}

    return EDSPixelData(
        elements=elements,
        counts=pixel_counts,
        weight_pct=wt_pct,
        atomic_pct=at_pct,
    )


@dataclass
class PhaseMatch:
    """Result of matching a composition to a phase."""
    phase_name: str
    score: float              # 0.0 (no match) to 1.0 (perfect match)
    expected_composition: Dict[str, float]  # At.% from phase definition


def suggest_phases(
    measured_at_pct: Dict[str, float],
    phase_definitions: Dict[str, Dict[str, float]],
    tolerance: float = 15.0,
) -> List[PhaseMatch]:
    """
    Suggest matching crystal phases based on measured atomic percent composition.

    Compares measured At.% with expected phase compositions using
    a simple distance metric. Only elements present in the phase
    definition are compared.

    Args:
        measured_at_pct: Measured At.% per element (e.g., {"Fe": 50.0, "O": 50.0})
        phase_definitions: Dict of phase_name -> {element: expected_at_pct}
            Example: {"Fe2O3": {"Fe": 40.0, "O": 60.0}}
        tolerance: Maximum allowed deviation per element in At.%

    Returns:
        List of PhaseMatch objects sorted by score (best first)
    """
    matches = []

    for phase_name, expected in phase_definitions.items():
        # Check if all expected elements are present in measurement
        all_present = all(
            elem in measured_at_pct for elem in expected
        )

        if not all_present:
            continue

        # Calculate score based on element-wise deviation
        deviations = []
        for elem, expected_pct in expected.items():
            measured_pct = measured_at_pct.get(elem, 0.0)
            deviation = abs(measured_pct - expected_pct)
            deviations.append(deviation)

        if not deviations:
            continue

        max_deviation = max(deviations)
        avg_deviation = sum(deviations) / len(deviations)

        # Skip if any element is too far off
        if max_deviation > tolerance * 2:
            continue

        # Score: 1.0 = perfect match, 0.0 = at tolerance limit
        score = max(0.0, 1.0 - (avg_deviation / tolerance))

        matches.append(PhaseMatch(
            phase_name=phase_name,
            score=score,
            expected_composition=expected,
        ))

    # Sort by score (highest first)
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


# Common Fe-bearing phase definitions (At.%)
# These are approximate stoichiometric compositions for phase pre-selection
COMMON_FE_PHASES: Dict[str, Dict[str, float]] = {
    "alpha-Fe (Ferrite/BCC Iron)": {"Fe": 100.0},
    "gamma-Fe (Austenite/FCC Iron)": {"Fe": 100.0},
    "Fe3C (Cementite)": {"Fe": 75.0, "C": 25.0},
    "Fe2O3 (Hematite)": {"Fe": 40.0, "O": 60.0},
    "Fe3O4 (Magnetite)": {"Fe": 42.9, "O": 57.1},
    "FeO (Wuestite)": {"Fe": 50.0, "O": 50.0},
    "FeS2 (Pyrite)": {"Fe": 33.3, "S": 66.7},
    "FeTiO3 (Ilmenite)": {"Fe": 20.0, "Ti": 20.0, "O": 60.0},
    "Fe2SiO4 (Fayalite)": {"Fe": 28.6, "Si": 14.3, "O": 57.1},
    "FeCr2O4 (Chromite)": {"Fe": 14.3, "Cr": 28.6, "O": 57.1},
    "Fe-3%Si (Electrical Steel)": {"Fe": 94.1, "Si": 5.9},
    "AISI 304 (Austenitic SS)": {"Fe": 68.0, "Cr": 19.0, "Ni": 10.0, "Mn": 2.0, "Si": 1.0},
    "AISI 316L (Austenitic SS)": {"Fe": 65.0, "Cr": 17.0, "Ni": 12.0, "Mo": 2.5, "Mn": 2.0},
    "Fe-Ni (Taenite)": {"Fe": 70.0, "Ni": 30.0},
    "FeAl (Iron Aluminide)": {"Fe": 50.0, "Al": 50.0},
    "Fe3Al": {"Fe": 75.0, "Al": 25.0},
}

COMMON_AL_PHASES: Dict[str, Dict[str, float]] = {
    "Al (Aluminum)": {"Al": 100.0},
    "Al2O3 (Corundum)": {"Al": 40.0, "O": 60.0},
    "Al2SiO5 (Sillimanite)": {"Al": 25.0, "Si": 12.5, "O": 62.5},
    "AlN (Aluminum Nitride)": {"Al": 50.0, "N": 50.0},
}

COMMON_SI_PHASES: Dict[str, Dict[str, float]] = {
    "Si (Silicon)": {"Si": 100.0},
    "SiO2 (Quartz)": {"Si": 33.3, "O": 66.7},
    "SiC (Silicon Carbide)": {"Si": 50.0, "C": 50.0},
}

# Al-Fe-Si intermetallic phases common in Al extrusion alloys (At.%)
COMMON_ALFE_PHASES: Dict[str, Dict[str, float]] = {
    "Al6Fe": {"Al": 85.7, "Fe": 14.3},
    "alpha-AlFeSi (Al8Fe2Si)": {"Al": 72.7, "Fe": 18.2, "Si": 9.1},
    "beta-AlFeSi (Al5FeSi)": {"Al": 71.4, "Fe": 14.3, "Si": 14.3},
    "Al13Fe4": {"Al": 76.5, "Fe": 23.5},
    "Al3Fe": {"Al": 75.0, "Fe": 25.0},
}

# Combined default phase library
DEFAULT_PHASE_LIBRARY: Dict[str, Dict[str, float]] = {
    **COMMON_FE_PHASES,
    **COMMON_AL_PHASES,
    **COMMON_SI_PHASES,
    **COMMON_ALFE_PHASES,
}


def filter_pixels_by_chemistry(
    counts: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    element: str,
    min_at_pct: float = 0.0,
    max_at_pct: float = 100.0,
    k_factors: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Create a boolean mask of pixels matching a chemistry criterion.

    Selects pixels where the At.% of a given element falls within
    [min_at_pct, max_at_pct]. Useful for selective indexing — e.g.,
    only index Fe-rich pixels.

    Args:
        counts: Dict mapping element symbol -> 1D counts array (flattened)
        n_rows: Number of rows in the scan grid
        n_cols: Number of columns in the scan grid
        element: Element symbol to filter on (e.g., "Fe")
        min_at_pct: Minimum At.% threshold (inclusive)
        max_at_pct: Maximum At.% threshold (inclusive)
        k_factors: Optional custom k-factors

    Returns:
        2D boolean mask (n_rows, n_cols) — True where criterion is met
    """
    # Convert all counts to At.%
    _, _, at_pct = counts_to_all(counts, k_factors)

    if element not in at_pct:
        logger.warning("Element '%s' not in EDS data, returning empty mask", element)
        return np.zeros((n_rows, n_cols), dtype=bool)

    elem_at = at_pct[element]
    # Reshape to 2D
    total = n_rows * n_cols
    if len(elem_at) < total:
        padded = np.zeros(total)
        padded[:len(elem_at)] = elem_at
        elem_at = padded

    elem_2d = elem_at[:total].reshape(n_rows, n_cols)
    mask = (elem_2d >= min_at_pct) & (elem_2d <= max_at_pct)
    return mask


def filter_pixels_by_phase(
    counts: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    phase_name: str,
    phase_library: Optional[Dict[str, Dict[str, float]]] = None,
    min_score: float = 0.5,
    k_factors: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Create a boolean mask of pixels matching a phase composition.

    For each pixel, computes the phase match score and selects pixels
    where the score exceeds min_score.

    Args:
        counts: Dict mapping element symbol -> 1D counts array (flattened)
        n_rows: Number of rows in the scan grid
        n_cols: Number of columns in the scan grid
        phase_name: Name of the phase in the library
        phase_library: Phase definitions dict; defaults to DEFAULT_PHASE_LIBRARY
        min_score: Minimum match score (0-1) to include pixel
        k_factors: Optional custom k-factors

    Returns:
        2D boolean mask (n_rows, n_cols) — True where phase matches
    """
    if phase_library is None:
        phase_library = DEFAULT_PHASE_LIBRARY

    if phase_name not in phase_library:
        logger.warning("Phase '%s' not in library, returning empty mask", phase_name)
        return np.zeros((n_rows, n_cols), dtype=bool)

    total = n_rows * n_cols
    mask = np.zeros(total, dtype=bool)

    for idx in range(total):
        row, col = divmod(idx, n_cols)
        pixel = get_pixel_composition(counts, row, col, n_cols, k_factors)
        matches = suggest_phases(pixel.atomic_pct, {phase_name: phase_library[phase_name]})
        if matches and matches[0].score >= min_score:
            mask[idx] = True

    return mask.reshape(n_rows, n_cols)


@dataclass
class PhaseRegionComposition:
    """Average EDS composition for a phase region in a consensus map."""
    phase_idx: int
    phase_name: str
    n_pixels: int
    avg_at_pct: Dict[str, float]
    avg_wt_pct: Dict[str, float]
    phase_matches: List[PhaseMatch]


def compute_phase_region_composition(
    counts: Dict[str, np.ndarray],
    phase_map: np.ndarray,
    phase_names: List[str],
    n_cols: int,
    k_factors: Optional[Dict[str, float]] = None,
    phase_library: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[PhaseRegionComposition]:
    """Compute average EDS composition for each phase region in a consensus map.

    Parameters
    ----------
    counts : dict
        Element symbol -> 1D counts array (flattened).
    phase_map : np.ndarray
        2D array (n_rows, n_cols) with phase indices (0, 1, ...) or -1.
    phase_names : list of str
        Phase display names, indexed by phase_map values.
    n_cols : int
        Number of columns in the scan grid.
    k_factors : dict, optional
        Custom k-factors; defaults to K_FACTORS_REL_SI.
    phase_library : dict, optional
        Phase definitions for suggest_phases(); defaults to DEFAULT_PHASE_LIBRARY.

    Returns
    -------
    list of PhaseRegionComposition
        One entry per phase (excluding unindexed pixels with -1).
    """
    if phase_library is None:
        phase_library = DEFAULT_PHASE_LIBRARY

    # Convert all counts to Wt.% and At.%
    _, wt_pct_arrays, at_pct_arrays = counts_to_all(counts, k_factors)

    flat_map = phase_map.ravel()
    results = []

    for phase_idx in range(len(phase_names)):
        pixels = np.where(flat_map == phase_idx)[0]
        if pixels.size == 0:
            results.append(PhaseRegionComposition(
                phase_idx=phase_idx,
                phase_name=phase_names[phase_idx],
                n_pixels=0,
                avg_at_pct={},
                avg_wt_pct={},
                phase_matches=[],
            ))
            continue

        # Average At.% and Wt.% over the region
        avg_at: Dict[str, float] = {}
        avg_wt: Dict[str, float] = {}
        for elem, arr in at_pct_arrays.items():
            valid = pixels[pixels < len(arr)]
            avg_at[elem] = float(np.mean(arr[valid])) if valid.size > 0 else 0.0
        for elem, arr in wt_pct_arrays.items():
            valid = pixels[pixels < len(arr)]
            avg_wt[elem] = float(np.mean(arr[valid])) if valid.size > 0 else 0.0

        # Phase suggestion based on average At.%
        matches = suggest_phases(avg_at, phase_library)

        results.append(PhaseRegionComposition(
            phase_idx=phase_idx,
            phase_name=phase_names[phase_idx],
            n_pixels=int(pixels.size),
            avg_at_pct=avg_at,
            avg_wt_pct=avg_wt,
            phase_matches=matches,
        ))

    return results
