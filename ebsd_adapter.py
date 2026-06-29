"""
EBSD Data Adapter
Converts unified_loader.EBSDData to kikuchipy.signals.EBSD format.

This adapter bridges the gap between the universal loader (which handles
multiple vendor formats with robust error handling) and kikuchipy's expected
EBSD signal structure.

IMPORTANT: Pattern Center (PC) Convention Handling
------------------------------------------------
Different EBSD vendors use different coordinate systems for the PC:
- EDAX/TSL: Origin at top-left, y increases downward
- Oxford: Origin at center, different axis conventions
- Bruker: Kikuchipy's native convention
- EMsoft: Uses pixel coordinates instead of fractions

This adapter automatically detects the vendor from DetectorData.vendor
and passes the correct 'convention' parameter to EBSDDetector, which
handles the conversion to kikuchipy's internal (Bruker) convention.

See: https://kikuchipy.org/en/stable/tutorials/reference_frames.html
"""

import numpy as np
from typing import Optional
from unified_loader import EBSDData
from kikuchipy.signals import EBSD
from kikuchipy.detectors import EBSDDetector
from hyperspy.signals import Signal2D


def ebsd_data_to_kikuchipy_signal(ebsd_data: EBSDData) -> EBSD:
    """
    Convert unified_loader.EBSDData to kikuchipy.signals.EBSD.

    This function:
    1. Creates an EBSDDetector from DetectorData
    2. Reshapes pattern arrays to kikuchipy's expected 4D format
    3. Sets up axes_manager with proper units and scales
    4. Returns a fully configured EBSD signal object

    Args:
        ebsd_data: EBSDData object from unified_loader

    Returns:
        kikuchipy.signals.EBSD signal compatible with all kikuchipy functions

    Note:
        - Pattern reshaping: unified_loader gives 3D (n_points, H, W),
          kikuchipy expects 4D (n_rows, n_cols, H, W)
        - PC handling: Uses mean PC for detector, per-point PC stored but
          not directly used by kikuchipy's EBSDDetector
        - Axes units: Converts to µm for navigation axes
    """
    # 1. Create EBSDDetector from DetectorData
    detector = _create_detector(ebsd_data.detector)

    # 2. Reshape patterns to 4D if needed
    patterns = _reshape_patterns(ebsd_data.patterns, ebsd_data.navigation)

    # 3. Create Signal2D with axes metadata
    sig2d = Signal2D(patterns)
    _setup_axes(sig2d, ebsd_data.navigation)

    # 4. Create EBSD signal
    ebsd_signal = EBSD(sig2d, detector=detector)

    # 5. Set static background if available
    if ebsd_data.static_background is not None:
        ebsd_signal.static_background = ebsd_data.static_background

    return ebsd_signal


def _create_detector(detector_data) -> EBSDDetector:
    """
    Create kikuchipy EBSDDetector from unified_loader DetectorData.

    IMPORTANT: Different vendors use different PC coordinate conventions!
    - EDAX (TSL): Origin at top-left, y increases downward
    - Oxford: Different origin and axis conventions
    - Bruker: Kikuchipy's native convention

    This function uses kikuchipy's built-in convention parameter to ensure
    PC values are correctly interpreted regardless of the source format.

    Args:
        detector_data: DetectorData from unified_loader

    Returns:
        Configured EBSDDetector instance with correct PC convention
    """
    # Map vendor names to kikuchipy convention names
    convention_map = {
        'tsl': 'tsl',      # EDAX uses TSL convention
        'edax': 'tsl',     # Alias
        'oxford': 'oxford',
        'bruker': 'bruker',
        'emsoft': 'emsoft',
    }

    # Get the convention, default to 'bruker' (kikuchipy native)
    vendor = getattr(detector_data, 'vendor', 'bruker')
    convention = convention_map.get(vendor.lower(), 'bruker')

    # Create detector with the appropriate convention
    # Kikuchipy will automatically convert PC values to its internal (Bruker) convention
    return EBSDDetector(
        shape=(detector_data.pattern_height, detector_data.pattern_width),
        px_size=55.0,  # Default pixel size in µm (typical EBSD scintillator)
        binning=detector_data.binning,
        tilt=detector_data.detector_tilt,  # Camera elevation angle (~10°)
        sample_tilt=detector_data.sample_tilt,  # Sample tilt from horizontal (~70°)
        azimuthal=detector_data.azimuthal,
        pc=(detector_data.pc_x, detector_data.pc_y, detector_data.pc_z),
        convention=convention,  # Tell kikuchipy which convention the PC values use
    )


def _reshape_patterns(patterns: np.ndarray, navigation) -> np.ndarray:
    """
    Reshape pattern array to kikuchipy's expected 4D format.

    unified_loader returns: (n_points, height, width) [3D]
    kikuchipy expects: (n_rows, n_cols, height, width) [4D]

    Args:
        patterns: Pattern array from unified_loader
        navigation: NavigationData with grid dimensions

    Returns:
        Reshaped 4D array
    """
    if patterns.ndim == 3:
        n_points, height, width = patterns.shape
        n_rows = navigation.n_rows
        n_cols = navigation.n_cols

        # Verify dimensions match
        if n_points != n_rows * n_cols:
            raise ValueError(
                f"Pattern count mismatch: {n_points} patterns but "
                f"grid is {n_rows}×{n_cols} = {n_rows * n_cols}"
            )

        # Reshape to 4D
        patterns = patterns.reshape(n_rows, n_cols, height, width)

    elif patterns.ndim == 4:
        # Already 4D, nothing to do
        pass

    else:
        raise ValueError(f"Unexpected pattern array dimensionality: {patterns.ndim}D")

    return patterns


def _setup_axes(sig2d: Signal2D, navigation) -> None:
    """
    Configure axes_manager for the Signal2D object.

    Sets navigation axes (y, x) with proper scales and units.

    Args:
        sig2d: Signal2D object to configure
        navigation: NavigationData with step sizes
    """
    # Y axis (rows)
    sig2d.axes_manager[0].name = "y"
    sig2d.axes_manager[0].scale = navigation.step_y
    sig2d.axes_manager[0].units = "µm"

    # X axis (columns)
    sig2d.axes_manager[1].name = "x"
    sig2d.axes_manager[1].scale = navigation.step_x
    sig2d.axes_manager[1].units = "µm"

    # Signal axes (pattern dimensions) are left as-is
    # They default to pixel units with scale=1


if __name__ == '__main__':
    """Test the adapter with sample files from Test_data/."""
    import sys
    from pathlib import Path
    from unified_loader import EBSDDataLoader

    # Resolve Test_data/ relative to this script so the smoke test works
    # on any machine without hand-editing paths (project requirement).
    test_data_dir = Path(__file__).resolve().parent / "Test_data"
    test_files = sorted(
        str(p) for p in test_data_dir.glob("*.h5*")
    ) if test_data_dir.is_dir() else []

    if not test_files:
        print(f"No H5/H5OINA files found in {test_data_dir}")
        sys.exit(0)

    for filepath in test_files:
        if not Path(filepath).exists():
            print(f"File not found: {filepath}")
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {Path(filepath).name}")
        print('='*60)

        # Load with unified_loader
        loader = EBSDDataLoader(filepath)
        ebsd_data = loader.load()

        print(f"Loaded EBSDData:")
        print(f"  Format: {ebsd_data.format}")
        print(f"  Vendor/Convention: {ebsd_data.detector.vendor}")
        print(f"  Patterns: {ebsd_data.patterns.shape} ({ebsd_data.pattern_type})")
        print(f"  Grid: {ebsd_data.navigation.n_rows}×{ebsd_data.navigation.n_cols}")
        print(f"  Raw PC: ({ebsd_data.detector.pc_x:.4f}, {ebsd_data.detector.pc_y:.4f}, {ebsd_data.detector.pc_z:.4f})")

        # Convert to kikuchipy signal
        try:
            ebsd_signal = ebsd_data_to_kikuchipy_signal(ebsd_data)
            print(f"\nConverted to kikuchipy EBSD signal:")
            print(f"  Signal shape: {ebsd_signal.data.shape}")
            print(f"  Detector: {ebsd_signal.detector}")
            print(f"  PC (Bruker/internal): {ebsd_signal.detector.pc}")
            print(f"  PC (TSL):    {ebsd_signal.detector.pc_tsl()}")
            print(f"  PC (Oxford): {ebsd_signal.detector.pc_oxford()}")
            print(f"  Axes: {ebsd_signal.axes_manager}")
            print("  ✓ Conversion successful!")
        except Exception as e:
            print(f"  ✗ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
