"""
Safe EBSD Loader with Automatic Fallback
Provides robust loading of EBSD HDF5 files with multiple vendor support.

This module implements a hybrid loading strategy:
1. Try kikuchipy's native loader first (fast, optimal for compatible files)
2. On failure, fall back to unified_loader (robust, handles missing fields)
3. Convert unified_loader output to kikuchipy format via adapter

This approach ensures maximum compatibility while maintaining performance
for well-formed files.

Also installs a tightly scoped runtime workaround for an upstream kikuchipy
bug that crashes on Oxford H5OINA files written by Aztec 6.2+; see
``_kikuchipy_oxford_camera_binning_workaround`` for details.
"""

import contextlib
import logging
import os
from pathlib import Path
from typing import Optional

# Lazy-load kikuchipy to avoid Numba JIT cache errors on startup
kp = None

def _kp():
    global kp
    if kp is None:
        # Disable Numba JIT if cache is broken (Python 3.13 compatibility)
        if not os.environ.get("NUMBA_DISABLE_JIT"):
            try:
                import kikuchipy as _kp_mod
                kp = _kp_mod
            except (OSError, FileNotFoundError):
                os.environ["NUMBA_DISABLE_JIT"] = "1"
                import kikuchipy as _kp_mod
                kp = _kp_mod
        else:
            import kikuchipy as _kp_mod
            kp = _kp_mod
    return kp

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _kikuchipy_oxford_camera_binning_workaround():
    """Patch ``OxfordH5EBSDReader.scan2dict`` so it tolerates Aztec H5OINA files
    that omit ``Camera Binning Mode`` from ``/1/EBSD/Header``.

    Upstream bug (verified in kikuchipy 0.11.3 on this machine, file
    ``kikuchipy/io/plugins/oxford_h5ebsd/_api.py`` around line 152):

        binning_str = header_group.get("Camera Binning Mode")    # -> None
        try:
            detector_kw["binning"] = int(binning_str.split("x")[0])
        except (IndexError, ValueError):                          # misses AttributeError
            ...

    Aztec 6.2+ writes ``Camera Mode`` + ``Camera Gain`` instead of
    ``Camera Binning Mode``; ``header_group.get(...)`` returns ``None`` and
    ``None.split(...)`` raises ``AttributeError`` which the bare except does
    not catch. Result: every Oxford H5OINA in this project crashes on
    ``kp.load`` and ``safe_loader`` silently falls through to
    ``unified_loader`` (eager) — defeating the T1 lazy-load work.

    This context manager monkey-patches the broken method for the duration
    of the load call only, then restores the original. Survives kikuchipy
    upgrades as long as ``OxfordH5EBSDReader.scan2dict`` keeps its name.

    TODO: submit upstream PR widening ``except (IndexError, ValueError)`` to
    also catch ``AttributeError`` / ``TypeError``.
    """
    from kikuchipy.io.plugins.oxford_h5ebsd import _api as _ox_api
    Reader = _ox_api.OxfordH5EBSDReader
    original = Reader.scan2dict

    import h5py
    import numpy as np
    from orix.crystal_map import CrystalMap
    from kikuchipy.detectors.ebsd_detector import EBSDDetector
    from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict

    def patched_scan2dict(self, group: "h5py.Group", lazy: bool = False) -> dict:
        # Body copied verbatim from kikuchipy 0.11.3
        # (kikuchipy/io/plugins/oxford_h5ebsd/_api.py:scan2dict) with ONE
        # change: the except clause around the ``binning`` extraction now
        # also catches AttributeError/TypeError so ``None.split(...)`` is
        # handled gracefully.
        header_group = _hdf5group2dict(group["EBSD/Header"], recursive=True)
        data_group = _hdf5group2dict(
            group["EBSD/Data"], data_dset_names=self.pattern_dataset_names
        )

        ny, nx = header_group["Y Cells"], header_group["X Cells"]
        sy, sx = header_group["Pattern Height"], header_group["Pattern Width"]
        dy, dx = header_group.get("Y Step", 1), header_group.get("X Step", 1)
        px_size = 1.0

        fname, title = self.get_metadata_filename_title(group.name)
        metadata = {
            "Acquisition_instrument": {
                "SEM": {
                    "beam_energy": header_group.get("Beam Voltage"),
                    "magnification": header_group.get("Magnification"),
                    "working_distance": header_group.get("Working Distance"),
                },
            },
            "General": {"original_filename": fname, "title": title},
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
        }
        scan_dict = {"metadata": metadata}

        data = self.get_data(group, data_shape=(ny, nx, sy, sx), lazy=lazy)
        scan_dict["data"] = data

        scan_dict["axes"] = self.get_axes_list((ny, nx, sy, sx), (dy, dx, px_size))

        scan_dict["original_metadata"] = {
            "manufacturer": self.manufacturer,
            "version": self.version,
        }
        scan_dict["original_metadata"].update(header_group)

        xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        scan_dict["xmap"] = xmap

        scan_dict["static_background"] = header_group.get(
            "Processed Static Background"
        )

        pc = np.column_stack(
            (
                data_group.get("Pattern Center X", 0.5),
                data_group.get("Pattern Center Y", 0.5),
                data_group.get("Detector Distance", 0.5),
            )
        )
        if pc.size > 3:
            pc = pc.reshape((ny, nx, 3))
        detector_kw = dict(
            shape=(sy, sx),
            pc=pc,
            sample_tilt=np.rad2deg(
                header_group.get("Tilt Angle", np.deg2rad(70))
            ),
            convention="oxford",
        )
        detector_tilt_euler = header_group.get("Detector Orientation Euler")
        try:
            detector_kw["tilt"] = np.rad2deg(detector_tilt_euler[1]) - 90
        except (IndexError, TypeError):
            logger.debug("Could not read detector tilt")
        binning_str = header_group.get("Camera Binning Mode")
        try:
            detector_kw["binning"] = int(binning_str.split("x")[0])
        except (IndexError, ValueError, AttributeError, TypeError):
            # PATCHED: also swallow AttributeError/TypeError when
            # binning_str is None (Aztec 6.2+ doesn't write the key).
            # kikuchipy's EBSDDetector default binning=1 will be used.
            logger.debug(
                "Camera Binning Mode missing from H5OINA header — "
                "kikuchipy default binning=1 will be used"
            )
        scan_dict["detector"] = EBSDDetector(**detector_kw)

        return scan_dict

    Reader.scan2dict = patched_scan2dict
    try:
        yield
    finally:
        Reader.scan2dict = original


def load_ebsd_safe(
    file_path: str,
    pattern_type: str = 'processed',
    verbose: bool = True
):
    """
    Safely load EBSD data with automatic fallback mechanism.

    Attempts to load using kikuchipy's native loader first. The Oxford
    H5OINA reader in kikuchipy has a known crash on Aztec 6.2+ files that
    omit ``Camera Binning Mode`` from the header — this is now worked
    around at runtime via ``_kikuchipy_oxford_camera_binning_workaround``,
    so the native (lazy) path succeeds for all Oxford files in this
    project. The fallback to ``unified_loader`` (eager) remains as a
    backstop for genuinely broken files.

    Args:
        file_path: Path to HDF5 file (.h5, .hdf5, .h5oina)
        pattern_type: 'processed' or 'raw' (for Oxford files with unified_loader)
        verbose: Print loading information to console

    Returns:
        kikuchipy.signals.EBSD signal object. When the kikuchipy native
        loader succeeds (the common case), this is a LazyEBSD (data is
        a dask array, materialised on demand). When the unified_loader
        fallback fires, the returned signal is eager (data is numpy).
        Callers that perform in-place writes to signal.data must check
        signal._lazy and call signal.compute() first if needed.

    Raises:
        RuntimeError: If both loaders fail, with details from both attempts

    Examples:
        >>> sig = load_ebsd_safe('data.h5oina')
        >>> sig = load_ebsd_safe('data.h5oina', pattern_type='raw', verbose=False)
    """
    file_path = str(file_path)
    file_name = Path(file_path).name

    # Decide lazy vs eager based on file size. The lazy chain is a massive
    # win on multi-GB files (27 GB H5OINA: 10 min → 7.7 s) because we don't
    # have to read every pattern into RAM at load time. But on smaller
    # files it's a regression: a 339 MB file loads in <3 s eager and every
    # subsequent pattern / overview / atlas access is instant from RAM,
    # while the lazy path defers ALL that disk I/O to first access — so
    # the user-visible "load is done" point shifts from "after 3 s" to
    # "after 3 min of overview computation". Threshold is configurable
    # via KIKUCHIPY_LAZY_THRESHOLD_BYTES; default 2 GB sits comfortably
    # below most workstation RAM and above all the small-file workflows
    # in this project.
    try:
        size_bytes = os.path.getsize(file_path)
    except OSError:
        # File missing / unreadable — let the native loader raise the
        # canonical error.
        size_bytes = None
    try:
        lazy_threshold = int(os.environ.get(
            "KIKUCHIPY_LAZY_THRESHOLD_BYTES", str(2 * 1024**3)
        ))
    except ValueError:
        lazy_threshold = 2 * 1024**3
    use_lazy = size_bytes is None or size_bytes >= lazy_threshold

    # Strategy 1: Try kikuchipy's native loader (fast path).
    # Wrapped in a context manager that monkey-patches kikuchipy's broken
    # Oxford H5OINA reader for the duration of this call. See
    # ``_kikuchipy_oxford_camera_binning_workaround`` above.
    try:
        if verbose:
            size_mb = (size_bytes or 0) / 1024 / 1024
            logger.info(
                "Loading %s with kikuchipy native loader (%s, %.1f MB)...",
                file_name, "lazy" if use_lazy else "eager", size_mb,
            )
        with _kikuchipy_oxford_camera_binning_workaround():
            sig = _kp().load(file_path, lazy=use_lazy)
        if verbose:
            logger.info("  Success with kikuchipy")
        return sig

    except Exception as e_kp:
        # Kikuchipy failed - log and try fallback
        if verbose:
            error_msg = str(e_kp)
            # Truncate very long error messages
            if len(error_msg) > 150:
                error_msg = error_msg[:150] + "..."
            logger.info("  kikuchipy failed: %s: %s", type(e_kp).__name__, error_msg)

        # Strategy 2: Try unified_loader with adapter (robust fallback)
        try:
            if verbose:
                # Detect format for informative message
                try:
                    import h5py
                    from unified_loader import detect_format
                    with h5py.File(file_path, 'r') as f:
                        fmt = detect_format(f)
                    logger.info("  Falling back to unified_loader (%s format)...", fmt)
                except Exception:
                    logger.info("  Falling back to unified_loader...")

            # Load with unified_loader
            from unified_loader import EBSDDataLoader
            from ebsd_adapter import ebsd_data_to_kikuchipy_signal
            loader = EBSDDataLoader(file_path)
            ebsd_data = loader.load(pattern_type=pattern_type)

            # Convert to kikuchipy signal
            sig = ebsd_data_to_kikuchipy_signal(ebsd_data)

            if verbose:
                logger.info("  Success with unified_loader")
                logger.info("    Format: %s", ebsd_data.format)
                logger.info("    Patterns: %s", ebsd_data.patterns.shape)
                logger.info("    Grid: %dx%d", ebsd_data.navigation.n_rows, ebsd_data.navigation.n_cols)

            return sig

        except Exception as e_unified:
            # Both loaders failed - raise comprehensive error
            error_msg = (
                f"Failed to load {file_name} with both loaders:\n"
                f"\n"
                f"  1. kikuchipy native loader:\n"
                f"     {type(e_kp).__name__}: {str(e_kp)}\n"
                f"\n"
                f"  2. unified_loader:\n"
                f"     {type(e_unified).__name__}: {str(e_unified)}\n"
                f"\n"
                f"File may be corrupted or in an unsupported format."
            )
            raise RuntimeError(error_msg) from e_unified


def get_loader_info(file_path: str) -> dict:
    """
    Get information about a file without loading all data.

    Useful for displaying file details or choosing loading options.

    Args:
        file_path: Path to HDF5 file

    Returns:
        Dictionary with file information:
        - format: 'EDAX', 'Oxford', or 'Unknown'
        - has_processed_patterns: bool (Oxford only)
        - has_raw_patterns: bool (Oxford only)
        - pattern_shape: tuple
        - root_key: str

    Example:
        >>> info = get_loader_info('data.h5oina')
        >>> print(f"Format: {info['format']}")
        >>> if info['has_raw_patterns']:
        >>>     sig = load_ebsd_safe('data.h5oina', pattern_type='raw')
    """
    from unified_loader import EBSDDataLoader
    loader = EBSDDataLoader(file_path)
    return loader.get_info()


if __name__ == '__main__':
    """Test the safe loader with sample files from Test_data/."""
    import sys

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
            print(f"\nFile not found: {filepath}")
            continue

        print(f"\n{'='*70}")
        print(f"Testing: {Path(filepath).name}")
        print('='*70)

        # Test loading
        try:
            sig = load_ebsd_safe(filepath, verbose=True)
            print(f"\n✓ Loading successful!")
            print(f"  Signal shape: {sig.data.shape}")
            print(f"  Signal dtype: {sig.data.dtype}")
            print(f"  Detector PC: {sig.detector.pc}")
            print(f"  Detector shape: {sig.detector.shape}")
            print(f"  Navigation: {sig.axes_manager.navigation_shape}")
            print(f"  Signal: {sig.axes_manager.signal_shape}")

        except Exception as e:
            print(f"\n✗ Loading failed:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()

        print()
