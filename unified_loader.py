"""
Universal HDF5 Loader for EBSD Data
Supports EDAX (.h5) and Oxford Instruments H5OINA (.h5oina) formats.

This module provides format-agnostic loading of EBSD patterns and metadata,
normalizing data to consistent units and structures.
"""

import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class DetectorData:
    """Detector geometry and pattern center data."""
    # Global PC (single value for entire scan)
    pc_x: float
    pc_y: float
    pc_z: float

    # Per-point PC (array for each measurement point)
    pc_per_point: np.ndarray  # Shape: (n_points, 3)
    has_per_point_pc: bool  # True if per-point PC is real (not replicated)

    # Geometry
    sample_tilt: float  # Degrees
    detector_tilt: float  # Degrees (camera elevation)
    azimuthal: float  # Degrees
    working_distance: float  # mm

    # Pattern dimensions
    pattern_width: int
    pattern_height: int
    binning: int = 1

    # Vendor/Convention info for PC conversion
    # 'tsl' = EDAX TSL, 'oxford' = Oxford Instruments, 'bruker' = Bruker/Kikuchipy native
    vendor: str = 'bruker'


@dataclass
class NavigationData:
    """Scan grid information."""
    n_rows: int
    n_cols: int
    step_x: float  # µm
    step_y: float  # µm
    x_positions: np.ndarray  # µm
    y_positions: np.ndarray  # µm


@dataclass
class EBSDData:
    """Complete EBSD dataset."""
    # Core data
    patterns: np.ndarray  # Shape: (n_points, height, width) or (n_rows, n_cols, height, width)
    pattern_type: str  # 'processed' or 'raw'

    # Metadata
    detector: DetectorData
    navigation: NavigationData

    # File info
    format: str  # 'EDAX' or 'Oxford'
    filepath: str

    # Optional
    static_background: Optional[np.ndarray] = None


def detect_format(h5file: h5py.File) -> str:
    """
    Detect the format of an HDF5 EBSD file.

    Returns:
        'EDAX', 'Oxford', or 'Unknown'
    """
    # Check for Manufacturer dataset/attribute
    if 'Manufacturer' in h5file:
        mfg = h5file['Manufacturer']
        if isinstance(mfg, h5py.Dataset):
            mfg_val = mfg[0] if mfg.shape else mfg[()]
            if isinstance(mfg_val, bytes):
                mfg_val = mfg_val.decode('utf-8')
            mfg_str = str(mfg_val)
        else:
            mfg_str = str(mfg)

        if 'EDAX' in mfg_str:
            return 'EDAX'
        if 'Oxford' in mfg_str:
            return 'Oxford'

    # Check root attributes
    if 'Manufacturer' in h5file.attrs:
        mfg = h5file.attrs['Manufacturer']
        if isinstance(mfg, bytes):
            mfg = mfg.decode('utf-8')
        if 'EDAX' in str(mfg):
            return 'EDAX'
        if 'Oxford' in str(mfg):
            return 'Oxford'

    # Fallback: check structure patterns
    for key in h5file.keys():
        if isinstance(h5file[key], h5py.Group):
            # Check for EDAX-specific structure
            ebsd_path = f'{key}/EBSD/Header'
            if ebsd_path in h5file:
                header = h5file[ebsd_path]
                if 'Pattern Center Calibration' in header:
                    return 'EDAX'

            # Check for Oxford-specific structure
            data_path = f'{key}/EBSD/Data'
            if data_path in h5file:
                data = h5file[data_path]
                if 'Processed Patterns' in data or 'Unprocessed Patterns' in data:
                    return 'Oxford'

    return 'Unknown'


def find_root_key(h5file: h5py.File, format_type: str) -> str:
    """
    Find the root group key containing EBSD data.

    EDAX: Uses sample name (e.g., 'HiGainNi')
    Oxford: Uses numeric index (e.g., '1')
    """
    for key in h5file.keys():
        if isinstance(h5file[key], h5py.Group):
            if f'{key}/EBSD' in h5file:
                return key

    raise ValueError("Could not find EBSD data group in file")


def _safe_read(h5obj, path: str, default=None):
    """Safely read a value from HDF5, returning default if not found."""
    try:
        if path in h5obj:
            val = h5obj[path][()]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            elif isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            return val
    except Exception:
        pass
    return default


class EBSDDataLoader:
    """
    Universal loader for EBSD HDF5 files.

    Supports:
    - EDAX OIM (.h5)
    - Oxford Instruments Aztec H5OINA (.h5oina)

    Usage:
        loader = EBSDDataLoader('path/to/file.h5')
        data = loader.load()
        # or for Oxford with pattern choice:
        data = loader.load(pattern_type='raw')
    """

    def __init__(self, filepath: str):
        self.filepath = str(filepath)
        self.format: Optional[str] = None
        self.root_key: Optional[str] = None
        self._h5file: Optional[h5py.File] = None

    def get_info(self) -> Dict[str, Any]:
        """
        Get file information without loading all data.
        Useful for displaying options to user before full load.
        """
        with h5py.File(self.filepath, 'r') as f:
            self.format = detect_format(f)
            self.root_key = find_root_key(f, self.format)

            info = {
                'format': self.format,
                'filepath': self.filepath,
                'root_key': self.root_key,
            }

            if self.format == 'Oxford':
                # Check available pattern types
                data_path = f'{self.root_key}/EBSD/Data'
                info['has_processed_patterns'] = f'{data_path}/Processed Patterns' in f
                info['has_raw_patterns'] = f'{data_path}/Unprocessed Patterns' in f

                # Get pattern shapes
                if info['has_processed_patterns']:
                    info['processed_shape'] = f[f'{data_path}/Processed Patterns'].shape
                if info['has_raw_patterns']:
                    info['raw_shape'] = f[f'{data_path}/Unprocessed Patterns'].shape

            elif self.format == 'EDAX':
                data_path = f'{self.root_key}/EBSD/Data/Pattern'
                if data_path in f:
                    info['pattern_shape'] = f[data_path].shape

            return info

    def load(self, pattern_type: str = 'processed') -> EBSDData:
        """
        Load EBSD data from file.

        Args:
            pattern_type: 'processed' or 'raw' (Oxford only, EDAX ignores this)

        Returns:
            EBSDData object with all loaded data
        """
        with h5py.File(self.filepath, 'r') as f:
            self.format = detect_format(f)

            if self.format == 'Unknown':
                raise ValueError(f"Unknown file format: {self.filepath}")

            self.root_key = find_root_key(f, self.format)

            # Load all components
            patterns, actual_type = self._load_patterns(f, pattern_type)
            detector = self._load_detector(f, len(patterns))
            navigation = self._load_navigation(f)
            static_bg = self._load_static_background(f)

            return EBSDData(
                patterns=patterns,
                pattern_type=actual_type,
                detector=detector,
                navigation=navigation,
                format=self.format,
                filepath=self.filepath,
                static_background=static_bg,
            )

    def _load_patterns(self, f: h5py.File, pattern_type: str) -> Tuple[np.ndarray, str]:
        """Load pattern data."""
        if self.format == 'EDAX':
            path = f'{self.root_key}/EBSD/Data/Pattern'
            patterns = f[path][:]
            return patterns, 'processed'  # EDAX only has one type

        else:  # Oxford
            data_path = f'{self.root_key}/EBSD/Data'

            if pattern_type == 'raw' and f'{data_path}/Unprocessed Patterns' in f:
                patterns = f[f'{data_path}/Unprocessed Patterns'][:]
                return patterns, 'raw'
            elif f'{data_path}/Processed Patterns' in f:
                patterns = f[f'{data_path}/Processed Patterns'][:]
                return patterns, 'processed'
            elif f'{data_path}/Unprocessed Patterns' in f:
                patterns = f[f'{data_path}/Unprocessed Patterns'][:]
                return patterns, 'raw'
            else:
                raise ValueError("No pattern data found in Oxford file")

    def _load_detector(self, f: h5py.File, n_points: int) -> DetectorData:
        """Load detector geometry and PC data."""
        if self.format == 'EDAX':
            return self._load_detector_edax(f, n_points)
        else:
            return self._load_detector_oxford(f)

    def _load_detector_edax(self, f: h5py.File, n_points: int) -> DetectorData:
        """Load EDAX detector data."""
        header = f[f'{self.root_key}/EBSD/Header']

        # Pattern Center
        pc_cal = header['Pattern Center Calibration']
        pc_x = float(pc_cal['x-star'][()])
        pc_y = float(pc_cal['y-star'][()])
        pc_z = float(pc_cal['z-star'][()])

        # Create replicated per-point PC (EDAX has global PC only)
        pc_per_point = np.tile([pc_x, pc_y, pc_z], (n_points, 1))

        # Geometry - Sample Tilt already in degrees for EDAX
        sample_tilt = _safe_read(header, 'Sample Tilt', 70.0)

        # Camera/Detector angles
        detector_tilt = _safe_read(header, 'Camera Elevation Angle', 0.0)
        azimuthal = _safe_read(header, 'Camera Azimuthal Angle', 0.0)

        # Working distance - EDAX might store in µm, convert to mm
        wd = _safe_read(header, 'Working Distance', 15.0)
        # If WD seems too small (< 1), it's likely in mm already
        # If WD seems too large (> 100), it's likely in µm
        if wd > 100:
            wd = wd / 1000.0  # Convert µm to mm

        # Pattern dimensions
        pattern_width = int(_safe_read(header, 'Pattern Width', 60))
        pattern_height = int(_safe_read(header, 'Pattern Height', 60))

        return DetectorData(
            pc_x=pc_x,
            pc_y=pc_y,
            pc_z=pc_z,
            pc_per_point=pc_per_point,
            has_per_point_pc=False,
            sample_tilt=float(sample_tilt),
            detector_tilt=float(detector_tilt),
            azimuthal=float(azimuthal),
            working_distance=float(wd),
            pattern_width=pattern_width,
            pattern_height=pattern_height,
            vendor='tsl',  # EDAX uses TSL convention
        )

    def _load_detector_oxford(self, f: h5py.File) -> DetectorData:
        """Load Oxford detector data."""
        header = f[f'{self.root_key}/EBSD/Header']
        data = f[f'{self.root_key}/EBSD/Data']

        # Per-point Pattern Center arrays
        pc_x_arr = data['Pattern Center X'][:]
        pc_y_arr = data['Pattern Center Y'][:]
        pc_z_arr = data['Detector Distance'][:]

        pc_per_point = np.column_stack([pc_x_arr, pc_y_arr, pc_z_arr])

        # Global PC (mean values)
        pc_x = float(np.mean(pc_x_arr))
        pc_y = float(np.mean(pc_y_arr))
        pc_z = float(np.mean(pc_z_arr))

        # Sample Tilt - Oxford stores in radians, convert to degrees
        sample_tilt_rad = _safe_read(header, 'Tilt Angle', 1.2217)  # ~70°
        sample_tilt = float(sample_tilt_rad) * 180.0 / np.pi

        # Detector orientation from Euler angles if available
        detector_euler = _safe_read(header, 'Detector Orientation Euler', None)
        if detector_euler is not None and len(detector_euler) >= 2:
            # Second Euler angle often corresponds to detector tilt
            detector_tilt = float(detector_euler[0, 1]) * 180.0 / np.pi if detector_euler.ndim > 1 else 0.0
        else:
            detector_tilt = 0.0

        azimuthal = 0.0  # Oxford doesn't typically store this separately

        # Working distance - Oxford stores in mm
        wd = _safe_read(header, 'Working Distance', 15.0)

        # Pattern dimensions
        pattern_width = int(_safe_read(header, 'Pattern Width', 156))
        pattern_height = int(_safe_read(header, 'Pattern Height', 128))

        return DetectorData(
            pc_x=pc_x,
            pc_y=pc_y,
            pc_z=pc_z,
            pc_per_point=pc_per_point,
            has_per_point_pc=True,
            sample_tilt=sample_tilt,
            detector_tilt=detector_tilt,
            azimuthal=azimuthal,
            working_distance=float(wd),
            pattern_width=pattern_width,
            pattern_height=pattern_height,
            vendor='oxford',  # Oxford Instruments convention
        )

    def _load_navigation(self, f: h5py.File) -> NavigationData:
        """Load scan grid information."""
        if self.format == 'EDAX':
            return self._load_navigation_edax(f)
        else:
            return self._load_navigation_oxford(f)

    def _load_navigation_edax(self, f: h5py.File) -> NavigationData:
        """Load EDAX navigation data."""
        header = f[f'{self.root_key}/EBSD/Header']
        data = f[f'{self.root_key}/EBSD/Data']

        n_rows = int(_safe_read(header, 'nRows', 1))
        n_cols = int(_safe_read(header, 'nColumns', 1))
        step_x = float(_safe_read(header, 'Step X', 1.0))
        step_y = float(_safe_read(header, 'Step Y', 1.0))

        x_pos = data['X Position'][:] if 'X Position' in data else np.arange(n_cols) * step_x
        y_pos = data['Y Position'][:] if 'Y Position' in data else np.arange(n_rows) * step_y

        return NavigationData(
            n_rows=n_rows,
            n_cols=n_cols,
            step_x=step_x,
            step_y=step_y,
            x_positions=x_pos,
            y_positions=y_pos,
        )

    def _load_navigation_oxford(self, f: h5py.File) -> NavigationData:
        """Load Oxford navigation data."""
        header = f[f'{self.root_key}/EBSD/Header']
        data = f[f'{self.root_key}/EBSD/Data']

        # Oxford uses X Cells / Y Cells
        n_cols = int(_safe_read(header, 'X Cells', 1))
        n_rows = int(_safe_read(header, 'Y Cells', 1))
        step_x = float(_safe_read(header, 'X Step', 1.0))
        step_y = float(_safe_read(header, 'Y Step', 1.0))

        x_pos = data['X'][:] if 'X' in data else np.arange(n_cols) * step_x
        y_pos = data['Y'][:] if 'Y' in data else np.arange(n_rows) * step_y

        return NavigationData(
            n_rows=n_rows,
            n_cols=n_cols,
            step_x=step_x,
            step_y=step_y,
            x_positions=x_pos,
            y_positions=y_pos,
        )

    def _load_static_background(self, f: h5py.File) -> Optional[np.ndarray]:
        """Load static background if available."""
        if self.format == 'Oxford':
            # Oxford stores static background in header
            bg_path = f'{self.root_key}/EBSD/Header/Processed Static Background'
            if bg_path in f:
                return f[bg_path][:]

            bg_path_raw = f'{self.root_key}/EBSD/Header/Unprocessed Static Background'
            if bg_path_raw in f:
                return f[bg_path_raw][:]

        # EDAX typically doesn't include static background in the file
        return None


# Convenience function for quick loading
def load_ebsd(filepath: str, pattern_type: str = 'processed') -> EBSDData:
    """
    Quick load function for EBSD data.

    Args:
        filepath: Path to HDF5 file (.h5 or .h5oina)
        pattern_type: 'processed' or 'raw' (Oxford only)

    Returns:
        EBSDData object
    """
    loader = EBSDDataLoader(filepath)
    return loader.load(pattern_type=pattern_type)


if __name__ == '__main__':
    # Test the loader
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
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
            if Path(filepath).exists():
                print(f"\n{'='*60}")
                print(f"Testing: {Path(filepath).name}")
                print('='*60)

                loader = EBSDDataLoader(filepath)

                # Get info first
                info = loader.get_info()
                print(f"Format: {info['format']}")
                print(f"Root key: {info['root_key']}")

                # Load data
                data = loader.load()

                print(f"\nPatterns:")
                print(f"  Shape: {data.patterns.shape}")
                print(f"  Type: {data.pattern_type}")
                print(f"  Dtype: {data.patterns.dtype}")

                print(f"\nDetector:")
                print(f"  PC: ({data.detector.pc_x:.4f}, {data.detector.pc_y:.4f}, {data.detector.pc_z:.4f})")
                print(f"  Has per-point PC: {data.detector.has_per_point_pc}")
                print(f"  Sample Tilt: {data.detector.sample_tilt:.1f}°")
                print(f"  Working Distance: {data.detector.working_distance:.2f} mm")
                print(f"  Pattern Size: {data.detector.pattern_width} x {data.detector.pattern_height}")

                print(f"\nNavigation:")
                print(f"  Grid: {data.navigation.n_rows} x {data.navigation.n_cols}")
                print(f"  Step: {data.navigation.step_x:.2f} x {data.navigation.step_y:.2f} µm")

                if data.static_background is not None:
                    print(f"\nStatic Background: {data.static_background.shape}")
