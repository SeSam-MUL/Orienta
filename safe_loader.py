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
import re
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


def _kikuchipy_needs_binning_patch() -> bool:
    """True for kikuchipy < 0.12, the versions whose Oxford reader crashes on
    a missing ``Camera Binning Mode``.

    requirements.txt asks for kikuchipy >= 0.13.1, so a source install never
    takes this branch. It stays for an installation that still has 0.11.x, for
    instance a packaged app whose package update did not complete.

    A version string without a leading ``major.minor`` answers False with a
    warning: raising here would happen inside the load and send every H5OINA
    to the eager fallback — the very failure this gate exists to prevent."""
    version = str(getattr(_kp(), "__version__", ""))
    m = re.match(r"(\d+)\.(\d+)", version)
    if m is None:
        logger.warning(
            "Cannot read the kikuchipy version %r; assuming >= 0.12 and "
            "leaving its Oxford reader unpatched", version)
        return False
    return (int(m.group(1)), int(m.group(2))) < (0, 12)


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

    Fixed upstream in kikuchipy 0.12.0 (``get_binning``, which also reads
    ``Camera Mode`` for Aztec >= 7.0). From 0.12 on this is a no-op: patching
    there would replace the newer reader with this 0.11.3 copy, and the copy
    imports ``kikuchipy.detectors.ebsd_detector``, a module 0.12 removed —
    the ImportError sent every H5OINA to the eager fallback, silently.
    """
    if not _kikuchipy_needs_binning_patch():
        yield
        return

    from kikuchipy.io.plugins.oxford_h5ebsd import _api as _ox_api
    Reader = _ox_api.OxfordH5EBSDReader
    original = Reader.scan2dict

    import h5py
    import numpy as np
    from orix.crystal_map import CrystalMap
    from kikuchipy.detectors import EBSDDetector
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


def _calibration_group_is_complete(group) -> bool:
    """True when an EDAX ``Pattern Center Calibration`` group holds a finite ``x-star``, ``y-star``
    and ``z-star``."""
    import numpy as np
    for key in ("x-star", "y-star", "z-star"):
        if key not in group:
            return False
        try:
            value = float(np.asarray(group[key][()]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return False
        if not np.isfinite(value):
            return False
    return True


def _header_has_pc(h5file) -> bool:
    """True when a manufacturer's own header of this .h5 states a pattern centre.

    EDAX: ``<scan>/EBSD/Header/Pattern Center Calibration`` with all three of ``x-star``,
    ``y-star`` and ``z-star`` finite: kikuchipy's reader takes each component the header lacks
    from its placeholder 0.5, so a group that holds two of the three is not a PC. Oxford H5OINA:
    ``<scan>/EBSD/Data/Pattern Center X`` (a per-pixel PC).
    """
    import h5py
    for key in h5file.keys():
        group = h5file[key]
        if not isinstance(group, h5py.Group):
            continue
        if "EBSD/Header/Pattern Center Calibration" in group:
            if _calibration_group_is_complete(group["EBSD/Header/Pattern Center Calibration"]):
                return True
            continue
        if "EBSD/Data/Pattern Center X" in group:
            return True
    return False


def _is_edax_file(h5file) -> bool:
    """True when the root says the file was written by EDAX software (or by an
    Orienta export in the EDAX layout, which states ``Manufacturer = EDAX``)."""
    try:
        value = h5file["Manufacturer"][()] if "Manufacturer" in h5file else h5file.attrs.get("Manufacturer")
        if isinstance(value, bytes):
            value = value.decode("utf-8", "ignore")
        return value is not None and "EDAX" in str(value)
    except Exception:  # noqa: BLE001
        return False


def _restore_pc_from_orienta_export(sig, file_path: str, verbose: bool = True) -> None:
    """Settle where the PC of a loaded .h5 comes from, when its header has none.

    An Orienta .h5 states the PC its run used in ``/Detector/pc`` (Bruker). An
    export of an EDAX .up1/.up2 made before the header carried the PC has the EDAX
    layout without ``Pattern Center Calibration``; kikuchipy then loads it at its
    placeholder, tilt 0, silently. The manufacturer's header always wins;
    ``/Detector/pc`` is used only when the header has none, only when it is a
    finite triple and only when ``/Detector/shape`` (if recorded) is this signal's
    own pattern shape (another detector's PC is not this one's).

    When no PC can be found - an export of a file that had none (``/Detector``
    says ``pc_missing``), or an EDAX .h5 whose header never had one - the placeholder
    stays, and the signal is tagged ``Signal.pc_source = "default"``, as an UP1
    without its .osc is, so the file is registered as having no PC instead of
    passing a made-up one off as a header value.
    """
    import h5py
    import numpy as np
    name = os.path.basename(file_path)
    try:
        applied = False
        edax = False
        with h5py.File(file_path, "r") as f:
            if _header_has_pc(f):
                # The header wins. If the export states another PC in /Detector/pc
                # (a PC typed or refined after the file was written), say so: it is
                # not applied, and the user would otherwise not know.
                if "Detector" in f and "pc" in f["Detector"] and not bool(
                        f["Detector"].attrs.get("pc_missing", False)):
                    exported = np.asarray(f["Detector/pc"][()], dtype=float).ravel()
                    now = np.asarray(sig.detector.pc, dtype=float).reshape(-1, 3).mean(axis=0)
                    if (exported.size == 3 and np.all(np.isfinite(exported))
                            and not np.allclose(exported, now, rtol=0, atol=1e-6)):
                        sig.metadata.set_item("Signal.export_pc_ignored",
                                              [float(v) for v in exported])
                        logger.warning(
                            "%s: /Detector/pc %s of this Orienta export differs from the PC in "
                            "the file's header %s; the header wins and /Detector/pc is not "
                            "applied.", name, tuple(round(float(v), 4) for v in exported),
                            tuple(round(float(v), 4) for v in now))
                return
            edax = _is_edax_file(f)
            exported_missing = "Detector" in f and bool(f["Detector"].attrs.get("pc_missing", False))
            pc = shape = None
            if "Detector" in f and "pc" in f["Detector"] and not exported_missing:
                pc = np.asarray(f["Detector/pc"][()], dtype=float).ravel()
                shape = f["Detector"].attrs.get("shape")
        if pc is not None and pc.size == 3 and np.all(np.isfinite(pc)):
            det = sig.detector
            if shape is not None and tuple(int(v) for v in np.asarray(shape).ravel()) != tuple(
                    int(v) for v in det.shape):
                logger.info("%s: /Detector/pc is for a %s detector, not %s: not applied",
                            name, tuple(np.asarray(shape).ravel()), tuple(det.shape))
            else:
                from kikuchipy.detectors import EBSDDetector
                sig.detector = EBSDDetector(
                    shape=tuple(int(v) for v in det.shape), px_size=det.px_size, binning=det.binning,
                    tilt=det.tilt, azimuthal=det.azimuthal, sample_tilt=det.sample_tilt,
                    pc=[float(v) for v in pc])
                sig.metadata.set_item("Signal.pc_source", "orienta_export")
                applied = True
                if verbose:
                    logger.info("%s: the header has no pattern centre; using /Detector/pc %s "
                                "from the Orienta export", name,
                                tuple(round(float(v), 4) for v in pc))
        if not applied and (edax or exported_missing):
            sig.metadata.set_item("Signal.pc_source", "default")
            logger.warning(
                "%s carries no pattern centre; kikuchipy's placeholder is in use. "
                "Calibrate the PC (PC refinement) before indexing.", name)
    except Exception:
        logger.debug("could not settle the PC of %s", file_path, exc_info=True)


def _is_oxford_layout(file_path: str) -> bool:
    """True when an HDF5 file has the Oxford layout (``Manufacturer`` Oxford, a
    ``/<scan>/EBSD/Data`` group) whatever its extension is called."""
    try:
        import h5py
        from unified_loader import detect_format
        with h5py.File(file_path, "r") as f:
            return detect_format(f) == "Oxford"
    except Exception:  # noqa: BLE001
        return False


def _load_oxford_by_layout(file_path: str, use_lazy: bool, verbose: bool, load_kwargs: Optional[dict] = None):
    """Read an Oxford-layout file with kikuchipy's Oxford reader, whatever its extension.
    ``load_kwargs`` (``scan_group_names``, ``processed``) is the chosen variant, see ``_resolve_variant``.

    ``kikuchipy.load`` picks its reader by the file EXTENSION: ``.h5oina`` goes to the
    Oxford reader, a ``.h5`` is offered to the EDAX / Bruker / kikuchipy / EMsoft h5ebsd
    readers only, none of which takes an Oxford file ("Could not read ..."). An Orienta
    rich export of an Oxford scan is such a file (a copy of the .h5oina with our groups
    added, saved under the name the user gave it, ``.h5`` by default). It then fell back to
    the eager unified loader, which keeps neither the camera tilt (4.3 deg became 0) nor
    the per-pixel PC map. The Oxford reader is called directly, as ``kikuchipy.load`` does
    for an ``.h5oina``.
    """
    from kikuchipy.io._io import _dict2signal
    from kikuchipy.io.plugins.oxford_h5ebsd import file_reader
    with _kikuchipy_oxford_camera_binning_workaround():
        dicts = file_reader(file_path, lazy=use_lazy, **(load_kwargs or {}))
    signals = [_dict2signal(d, lazy=use_lazy) for d in dicts]
    sig = signals[0] if len(signals) == 1 else signals
    if verbose:
        logger.info("  Success with kikuchipy's Oxford reader (file not named .h5oina)")
    return sig


#: Formats whose readers have no pattern centre to read: kikuchipy attaches its placeholder
#: (0.5, 0.5, 0.5) and nothing in the file says otherwise. NORDIF ``Pattern.dat`` and ``Setting.txt``
#: (calibration patterns), Oxford ``.ebsp`` and loose pattern images.
_PC_LESS_SUFFIXES = (".dat", ".txt", ".ebsp", ".tif", ".tiff", ".bmp", ".png")


def _tag_pc_less_formats(sig, file_path: str) -> None:
    """Mark a signal whose format carries no pattern centre as ``Signal.pc_source = "default"``.

    ``CalibrationStore.register`` turns that tag into ``pc_source == "missing"``, which every guard
    that refuses to work from a PC nobody supplied tests for. Before this, only an UP file without
    metadata and an EDAX .h5 without a header PC were tagged: a NORDIF or Oxford ``.ebsp`` scan was
    registered as having a measured PC and indexed at (0.5, 0.5, 0.5). A tag a loader set already
    (``osc``, ``apex``) is a statement about this very file and stays."""
    if Path(file_path).suffix.lower() not in _PC_LESS_SUFFIXES:
        return
    try:
        if sig.metadata.get_item("Signal.pc_source", None) is not None:
            return
        sig.metadata.set_item("Signal.pc_source", "default")
    except Exception as exc:
        # A PC-less file that cannot be tagged would register as having a measured PC: say so.
        logger.warning(
            "could not mark %s as having no pattern centre (%s: %s); it may be treated as having "
            "a measured PC. Check the pattern centre before indexing.",
            Path(file_path).name, type(exc).__name__, exc, exc_info=True)
        return
    logger.warning(
        "%s carries no pattern centre; kikuchipy's placeholder (0.5, 0.5, 0.5) is in use. "
        "Calibrate the PC (PC refinement) before indexing.", Path(file_path).name)


def _resolve_variant(file_path: str, variant):
    """Check a chosen variant against the file and say how to load it.

    Returns ``(resolved, load_kwargs)``: the variant with its measurement filled in, and the
    keyword arguments that make kikuchipy read exactly that scan group and pattern source.
    The group is always named (kikuchipy's own "first group" follows HDF5 key order: 1, 10, 2).
    ``processed`` is passed for Oxford files only, the one format with two pattern datasets.
    Raises ``LoadChoiceError`` when the file does not hold the variant."""
    from backend.api.services.load_choices import LoadVariant, pattern_dataset_path
    scan, _, dataset = pattern_dataset_path(file_path, variant).partition("/EBSD/Data/")
    load_kwargs = {"scan_group_names": [scan]}
    if dataset in ("Processed Patterns", "Unprocessed Patterns"):
        load_kwargs["processed"] = dataset == "Processed Patterns"
    return LoadVariant(scan, variant.pattern_source), load_kwargs


def _with_variant(sig, file_path: str, resolved):
    """Attach what belongs to the loaded variant to ``sig``: the static background of its own
    pattern source (kikuchipy reads the processed one whatever was loaded; a source without one has
    none, never the other's), and ``Orienta.load_variant``, which the route registers and drops
    (``variant_registry.register_from_signal``). Without a variant ``sig`` is returned unchanged."""
    if resolved is None:
        return sig
    from backend.api.services.load_choices import PROCESSED, UNPROCESSED
    if resolved.pattern_source in (PROCESSED, UNPROCESSED):
        import h5py
        import numpy as np
        kind = "Processed" if resolved.pattern_source == PROCESSED else "Unprocessed"
        path = f"{resolved.measurement}/EBSD/Header/{kind} Static Background"
        with h5py.File(file_path, "r") as f:
            background = np.asarray(f[path][()]) if path in f else None
        if background is None:
            sig._static_background = None     # the property setter takes an array only
        else:
            sig.static_background = background
    sig.metadata.set_item("Orienta.load_variant", resolved.to_dict())
    return sig


def _check_fallback_variant(loader, ebsd_data, resolved, file_name: str) -> None:
    """Refuse an eager-fallback load that did not read the chosen variant. ``unified_loader``
    takes the first scan group of the file and, for Oxford, the unprocessed patterns only when
    asked ('raw'); it cannot be told another group."""
    from backend.api.services.load_choices import LoadChoiceError
    wanted = {"processed": "processed", "unprocessed": "raw"}.get(resolved.pattern_source)
    if loader.root_key != resolved.measurement or (wanted and ebsd_data.pattern_type != wanted):
        raise LoadChoiceError(
            "loadVariantNeedsNativeReader",
            f"{file_name}: measurement {resolved.measurement!r} ({resolved.pattern_source} patterns) "
            "can only be read by kikuchipy's reader, which failed on this file")


def load_ebsd_safe(
    file_path: str,
    pattern_type: str = 'processed',
    verbose: bool = True,
    apex: Optional[dict] = None,
    variant=None,
):
    """Load an EBSD file (see ``_load_ebsd_safe_impl``) and say whether it carries a pattern centre."""
    sig = _load_ebsd_safe_impl(file_path, pattern_type, verbose, apex, variant)
    _tag_pc_less_formats(sig, str(file_path))
    from backend.api.services.load_choices import note_file_source
    note_file_source(sig, str(file_path), variant)
    return sig


def _load_ebsd_safe_impl(
    file_path: str,
    pattern_type: str = 'processed',
    verbose: bool = True,
    apex: Optional[dict] = None,
    variant=None,
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
        apex: ``{"h5_path": ..., "measurement_id": ...}``, the metadata file and measurement the
            user chose for an EDAX ``.up1``/``.up2``. Without it a UP file looks for an APEX
            ``.h5`` in its own folder that names it by GUID (``edax_apex``).
        variant: a ``load_choices.LoadVariant``, the scan group and pattern source to read from an
            HDF5 file. Without it the first group's processed patterns are read, as before. With
            it the group and source are named to kikuchipy, the static background is that of the
            source, and the signal carries ``Orienta.load_variant``. A variant the file does not
            hold raises ``load_choices.LoadChoiceError``; the eager fallback, which can read the
            first group only, raises it too rather than load another one. Formats read before this
            point (NORDIF, UP1/UP2, EDAX hex) have one variant and ignore it.

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

    # An EDAX APEX metadata file lists measurements and holds no patterns. Neither of the loaders
    # below can read it ("Could not read ..." / a 1 x 1 grid); the load dialog is how it is opened.
    if Path(file_path).suffix.lower() in (".h5", ".hdf5"):
        from edax_apex import ApexError, is_apex_file
        if is_apex_file(file_path):
            raise ApexError(
                "apexNeedsMeasurement",
                f"{file_name} is an EDAX APEX metadata file: it lists measurements but holds no "
                "patterns. Choose a measurement in the load dialog; its pattern file (.up1/.up2) "
                "is then loaded with this file's settings.")

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

    # A NORDIF ``Setting.txt`` stands for the calibration patterns beside it. kikuchipy returns
    # them as one-dimensional navigation; the PC page needs them as a scan of one row.
    if Path(file_path).suffix.lower() == ".txt":
        return _load_nordif_calibration(file_path, verbose)

    # EDAX UP1/UP2 raw-pattern files take a dedicated path: kikuchipy reads the
    # patterns natively, but a version-1 header carries no map grid (so the
    # signal would come back as a flat 1-D navigation) and neither v1 nor v3
    # carries the real µm step size. We recover both the grid and the step from
    # the companion .osc sidecar and hand them to kikuchipy. There is no
    # unified_loader fallback for this format, so failures are raised directly
    # with a clear message rather than falling through to a confusing HDF5 error.
    from edax_up1 import is_edax_up_file
    if is_edax_up_file(file_path):
        return _load_edax_up(file_path, use_lazy, verbose, apex)

    # EDAX scans acquired on a hexagonal grid cannot be handed to either
    # loader as they are: kikuchipy refuses HexGrid outright, and the
    # unified_loader trips over the padding EDAX writes to keep its point
    # arrays rectangular ("46648 patterns but grid is 397x118"). We resample
    # them onto a square grid first — nearest neighbour, exactly the way OIM
    # Analysis does it, verified against a square file OIM itself produced.
    # An Orienta .h5 that holds a hexagonal scan natively (index order, no padding, its own
    # geometry in the header) is not an OIM file: it loads as the index rectangle with its
    # geometry attached, before the OIM test below, which is also True for it. Its one variant
    # is the default; a variant the file does not hold is refused as for any other .h5.
    from backend.api.services.orienta_hex_h5 import is_native_hex_file, load_native_hex
    if is_native_hex_file(file_path):
        native_variant = _resolve_variant(file_path, variant)[0] if variant is not None else None
        sig = load_native_hex(file_path, lazy=use_lazy, verbose=verbose)
        _restore_pc_from_orienta_export(sig, file_path, verbose)
        return _with_variant(sig, file_path, native_variant)

    from edax_hex import is_edax_hex_file, load_edax_hex
    if is_edax_hex_file(file_path):
        return load_edax_hex(file_path, lazy=use_lazy, verbose=verbose)

    from backend.api.services.load_choices import LoadChoiceError
    resolved, variant_kwargs = _resolve_variant(file_path, variant) if variant is not None else (None, {})

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
            sig = _kp().load(file_path, lazy=use_lazy, **variant_kwargs)
        if verbose:
            logger.info("  Success with kikuchipy")
        _restore_pc_from_orienta_export(sig, file_path, verbose)

    except Exception as e_kp:
        # Kikuchipy failed - log and try fallback
        if file_name.lower().endswith(".h5oina"):
            # Every H5OINA is meant to load natively (lazily). A fall-through
            # here costs minutes and the whole file in RAM, so it is never
            # only an INFO line: that is how a kikuchipy upgrade once turned
            # the fast path off for every installation without a trace.
            logger.warning(
                "kikuchipy's native loader failed on %s (%s: %s); retrying "
                "with an explicit scan group, and if that fails too, falling "
                "back to the eager loader, which reads the whole file into "
                "memory", file_name, type(e_kp).__name__, str(e_kp)[:150])
        if verbose:
            error_msg = str(e_kp)
            # Truncate very long error messages
            if len(error_msg) > 150:
                error_msg = error_msg[:150] + "..."
            logger.info("  kikuchipy failed: %s: %s", type(e_kp).__name__, error_msg)

        # Strategy 1b: retry with an explicit scan-group name.
        # kikuchipy's EDAX h5ebsd reader picks the alphabetically-first
        # non-metadata top group as "the scan", which breaks on our rich .h5
        # exports (they add /Detector, /Documentation, /Indexing at root, all
        # sorting before /Scan1) -> KeyError on EBSD/Header. Detect the real
        # EBSD scan group(s) (those with EBSD/Data + EBSD/Header) and name them
        # explicitly. Fixes re-loading any file we've already exported.
        # A chosen variant named its group in the first call already: the retry would repeat it.
        try:
            import h5py
            with h5py.File(file_path, "r") as _f:
                scans = [k for k in _f.keys()
                         if isinstance(_f[k], h5py.Group)
                         and "EBSD/Data" in _f[k] and "EBSD/Header" in _f[k]]
            if scans and resolved is None:
                if verbose:
                    logger.info("  Retrying kikuchipy with explicit scan group(s) %s", scans)
                with _kikuchipy_oxford_camera_binning_workaround():
                    sig = _kp().load(file_path, lazy=use_lazy,
                                     scan_group_names=scans[0] if len(scans) == 1 else scans)
                if verbose:
                    logger.info("  Success with kikuchipy (explicit scan group)")
                _restore_pc_from_orienta_export(sig, file_path, verbose)
                return sig
        except Exception:
            logger.debug("explicit-scan-group retry failed for %s", file_name, exc_info=True)

        # Strategy 1c: an Oxford-layout file under another extension (see
        # ``_load_oxford_by_layout``): kikuchipy's Oxford reader, called directly.
        if not file_name.lower().endswith(".h5oina") and _is_oxford_layout(file_path):
            try:
                sig = _load_oxford_by_layout(file_path, use_lazy, verbose, variant_kwargs)
                _restore_pc_from_orienta_export(sig, file_path, verbose)
            except Exception:
                logger.debug("Oxford reader on %s failed", file_name, exc_info=True)
            else:
                # outside the try: an error of the variant step is the load's error, not a reason to
                # try the next reader (which would read something else, or the whole file eagerly)
                return _with_variant(sig, file_path, resolved)

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
            if resolved is not None and resolved.pattern_source == "unprocessed":
                pattern_type = "raw"
            ebsd_data = loader.load(pattern_type=pattern_type)
            if resolved is not None:
                _check_fallback_variant(loader, ebsd_data, resolved, file_name)

            # Convert to kikuchipy signal
            sig = ebsd_data_to_kikuchipy_signal(ebsd_data)
            # An Orienta export that reached this branch has its own /Detector/pc: the
            # header-wins rule and its hint apply here too, not only on the kikuchipy paths.
            _restore_pc_from_orienta_export(sig, file_path, verbose)

            if verbose:
                logger.info("  Success with unified_loader")
                logger.info("    Format: %s", ebsd_data.format)
                logger.info("    Patterns: %s", ebsd_data.patterns.shape)
                logger.info("    Grid: %dx%d", ebsd_data.navigation.n_rows, ebsd_data.navigation.n_cols)

        except LoadChoiceError:
            raise
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
        else:
            return _with_variant(sig, file_path, resolved)

    else:
        # Strategy 1 succeeded. Outside its try: an error of the variant step surfaces as it is
        # instead of sending the load to the eager fallback.
        return _with_variant(sig, file_path, resolved)


def _calibration_scan_to_grid(scan: dict) -> dict:
    """Give the one-dimensional navigation of NORDIF calibration patterns a grid of one row.

    kikuchipy returns the patterns of ``Setting.txt`` as ``(N, sy, sx)`` with a single navigation
    axis. Everything in the application that reads a pattern does ``signal.data[row, col]``, so
    "row 0" of that array was a line of ONE pattern and the PC page could neither show nor refine
    a calibration pattern. As a ``(1, N, sy, sx)`` array they are an ordinary scan of one row.
    The navigation steps stay undefined: the patterns are not neighbours."""
    import numpy as np
    data = np.asarray(scan["data"])
    if data.ndim != 3:
        raise ValueError(f"calibration patterns should be (N, rows, cols), got {data.shape}")
    n, sy, sx = data.shape
    out = dict(scan)
    out["data"] = data.reshape(1, n, sy, sx)
    out["axes"] = [
        {"size": 1, "index_in_array": 0, "name": "y", "scale": 1.0, "offset": 0.0},
        {"size": n, "index_in_array": 1, "name": "x", "scale": 1.0, "offset": 0.0},
        {"size": sy, "index_in_array": 2, "name": "dy", "scale": 1.0, "offset": 0.0, "units": "um"},
        {"size": sx, "index_in_array": 3, "name": "dx", "scale": 1.0, "offset": 0.0, "units": "um"},
    ]
    return out


class NordifSettingError(ValueError):
    """A NORDIF ``Setting.txt`` that cannot be opened, worded for the user. ``code`` is what the
    interface translates on; the text names files, never the folders above them."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _load_nordif_calibration(file_path: str, verbose: bool):
    """Open the calibration patterns of a NORDIF ``Setting.txt`` as a scan of one row.

    Any ``.txt`` is routed here, so a file that is not a NORDIF settings file (or one whose
    calibration images are missing) is refused by name instead of with kikuchipy's bare
    ``IndexError`` / ``FileNotFoundError``."""
    from kikuchipy.io._io import _dict2signal
    from kikuchipy.io.plugins.nordif_calibration_patterns._api import file_reader
    if not Path(file_path).is_file():
        raise NordifSettingError("nordifSettingMissing",
                                 f"{Path(file_path).name}: the file does not exist")
    try:
        raw = file_reader(file_path)[0]
    except Exception as exc:
        # kikuchipy's own text names the missing bitmap by its full path
        from edax_apex import without_paths
        raise NordifSettingError(
            "nordifCalibrationUnreadable",
            f"{Path(file_path).name} is not a NORDIF Setting.txt, or the calibration patterns "
            f"it lists cannot be read ({type(exc).__name__}: {without_paths(exc, file_path)})"
        ) from exc
    scan = _calibration_scan_to_grid(raw)
    sig = _dict2signal(scan, lazy=False)
    if verbose:
        logger.info("Loaded %d NORDIF calibration patterns from %s",
                    scan["data"].shape[1], Path(file_path).name)
    return sig


def _load_edax_up(file_path: str, use_lazy: bool, verbose: bool, apex: Optional[dict] = None):
    """Load an EDAX UP1/UP2 file with the metadata of its APEX measurement, else its .osc sidecar.

    See ``load_ebsd_safe`` for why this is a separate path. An APEX measurement (named by the
    user, or found in the folder by the GUID in the file's header) gives the grid, the steps,
    the pattern centre and the camera angles, hexagonal or square. Otherwise: returns a
    kikuchipy EBSD signal with a correct 2-D navigation (v1 files reshaped from
    the .osc grid) and, where the .osc provides it, the real µm step size on
    the navigation axes so the map scale bar is correct. A hexagonal file without APEX
    metadata is refused (``EdaxHexUpError``).
    """
    from edax_apex import load_edax_up_with_apex, measurement_for_pattern_file, read_up_header
    from edax_up1 import read_up1_header, resolve_up1_geometry

    file_name = Path(file_path).name
    measurement, how = measurement_for_pattern_file(file_path, apex)
    if measurement is not None:
        return load_edax_up_with_apex(
            file_path, read_up1_header(file_path), measurement, how, use_lazy, verbose)

    # Without metadata too, a header that cannot be read is refused with its code, by file name.
    read_up_header(file_path)
    geom = resolve_up1_geometry(file_path)

    if geom.version == 1 and geom.nav_shape is None:
        raise RuntimeError(
            f"Cannot determine the scan grid for {file_name}: it is a "
            f"version-1 UP file (no grid in its header) and no usable .osc "
            f"sidecar with matching point count was found next to it. Place "
            f"the matching .osc file in the same folder and try again."
        )

    load_kwargs = {"lazy": use_lazy}
    if geom.nav_shape is not None:
        load_kwargs["nav_shape"] = geom.nav_shape

    if verbose:
        logger.info(
            "Loading EDAX %s (v%d, %s) with grid %s, step %s µm",
            file_name, geom.version, "lazy" if use_lazy else "eager",
            geom.nav_shape or "from-header", geom.step_yx,
        )

    sig = _kp().load(file_path, **load_kwargs)

    # Apply the real pattern centre from the .osc (EDAX/TSL xstar/ystar/zstar).
    # A UP1/UP2 file carries NO PC, so kikuchipy attaches a placeholder
    # (0.5, 0.5, 0.5) — which makes indexing badly wrong (the PC is what
    # indexing stands or falls on). The .osc has the real PC; a joint
    # orientation+PC refinement confirmed these values (~2.5x the match NCC vs
    # the default). When the .osc has no usable PC we KEEP the default but tag
    # the signal so the UI can warn the user loudly.
    pc_source = "default"
    if geom.pc_edax is not None:
        try:
            from edax_up1 import tsl_detector
            old = sig.detector
            # The detector was rebuilt from the shape and the sample tilt alone, so a camera
            # tilt, pixel size or binning the reader had set was dropped on the way.
            sig.detector = tsl_detector(
                old.shape, geom.pc_edax,
                sample_tilt=getattr(old, "sample_tilt", 70.0),
                tilt=getattr(old, "tilt", 0.0),
                azimuthal=getattr(old, "azimuthal", 0.0),
                px_size=getattr(old, "px_size", 1.0),
                binning=getattr(old, "binning", 1),
            )
            pc_source = "osc"
            if verbose:
                logger.info("Applied .osc pattern centre (TSL) %s to %s",
                            tuple(round(v, 4) for v in geom.pc_edax), file_name)
        except Exception:
            logger.warning("Could not apply .osc PC to %s — keeping default PC",
                           file_name, exc_info=True)
    if pc_source == "default":
        logger.warning(
            "No pattern centre found for %s — indexing will use kikuchipy's "
            "placeholder PC (0.5, 0.5, 0.5), which is almost certainly wrong. "
            "Calibrate the PC (PC refinement) before trusting the indexing.",
            file_name)
    try:
        sig.metadata.set_item("Signal.pc_source", pc_source)
    except Exception:
        logger.debug("Could not tag pc_source on %s metadata", file_name, exc_info=True)

    # Override the navigation-axis scale with the real .osc step so the map
    # scale bar reads correctly. kikuchipy orders navigation axes as [y, x];
    # geom.step_yx is (dy, dx) to match. Guarded so a signal with an
    # unexpected axis layout never breaks the load.
    if geom.step_yx is not None:
        dy, dx = geom.step_yx
        try:
            nav_axes = sig.axes_manager.navigation_axes
            if len(nav_axes) == 2:
                # navigation_axes is (x, y) in hyperspy's fast-index-first order
                for ax in nav_axes:
                    ax.scale = dx if ax.name == "x" else dy
                    ax.units = "um"
            elif len(nav_axes) == 1:
                nav_axes[0].scale = dx
                nav_axes[0].units = "um"
        except Exception:
            logger.warning("Could not apply .osc step to %s navigation axes",
                           file_name, exc_info=True)

    if verbose:
        logger.info("  Success with EDAX UP reader")
    return sig


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


def probe_ebsd_content(file_path) -> dict:
    """Report what an HDF5 acquisition actually contains, without loading it.

    Aztec writes "Elementverteilungsdaten" acquisitions that carry EDS maps and
    SE/FSE images but **no** ``/<n>/EBSD`` group at all. Both loaders reject
    those outright ("no top groups with subgroup name 'EBSD'"), so the file
    could not be opened even though the EDS viewer needs none of the EBSD
    machinery. This probe lets the caller take an EDS-only route instead of
    failing, and lets it say WHY there are no pattern views.

    Cheap: opens the file read-only and reads dataset *shapes*, never data.

    Returns
    -------
    dict
        ``has_patterns``, ``has_eds``, ``has_electron_images``, ``eds_only``,
        ``eds_elements``, ``electron_images``, ``n_patterns``, ``error``.
        Every field is safe to read even when the file is not HDF5 at all.
    """
    from pathlib import Path as _Path

    info = {
        "has_patterns": False,
        "has_eds": False,
        "has_electron_images": False,
        "eds_only": False,
        "eds_elements": [],
        "electron_images": [],
        "n_patterns": 0,
        "error": None,
    }

    path = _Path(file_path)
    if path.suffix.lower() not in (".h5", ".hdf5", ".h5oina"):
        return info

    try:
        import h5py
    except ImportError as e:            # pragma: no cover - h5py is a hard dep
        info["error"] = str(e)
        return info

    try:
        with h5py.File(str(path), "r") as f:
            for top in f.keys():
                grp = f.get(top)
                if not isinstance(grp, h5py.Group):
                    continue

                ebsd_data = grp.get("EBSD/Data")
                if isinstance(ebsd_data, h5py.Group):
                    for key in ("Processed Patterns", "Unprocessed Patterns",
                                "RawPatterns", "Pattern", "patterns"):
                        ds = ebsd_data.get(key)
                        # A zero-length patterns dataset is not usable data.
                        if isinstance(ds, h5py.Dataset) and ds.shape and ds.shape[0] > 0:
                            info["has_patterns"] = True
                            info["n_patterns"] = max(info["n_patterns"], int(ds.shape[0]))
                            break

                wi = grp.get("EDS/Data/Window Integral")
                if isinstance(wi, h5py.Group) and len(wi.keys()):
                    info["has_eds"] = True
                    for el in wi.keys():
                        if el not in info["eds_elements"]:
                            info["eds_elements"].append(str(el))
                elif isinstance(grp.get("EDS/Data"), h5py.Group):
                    info["has_eds"] = True

                ei = grp.get("Electron Image/Data")
                if isinstance(ei, h5py.Group) and len(ei.keys()):
                    info["has_electron_images"] = True
                    for img in ei.keys():
                        if img not in info["electron_images"]:
                            info["electron_images"].append(str(img))
    except Exception as e:
        info["error"] = str(e)
        return info

    info["eds_only"] = (
        not info["has_patterns"]
        and (info["has_eds"] or info["has_electron_images"])
    )
    return info


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
