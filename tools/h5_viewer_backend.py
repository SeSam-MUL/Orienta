"""
HDF5 Viewer Backend — Pure data-logic, no PyQt5.

Contains H5OINADataExtractor for reading Oxford H5OINA and EDAX HDF5 files.
Separated from GUI code so that GUI improvements (FEAT-28+) can find and
improve the viewer via gui/h5_viewer_gui.py.
"""

import logging
import h5py
import numpy as np

logger = logging.getLogger(__name__)


class H5OINADataExtractor:
    """
    Helper class for extracting data from H5OINA/HDF5 files with lazy loading.

    Supports:
    - Oxford H5OINA format (patterns, EDS, electron images)
    - EDAX HDF5 format (patterns only)

    Key feature: Lazy loading - patterns are loaded one at a time, not all at once.
    """

    def __init__(self, h5file, format_type):
        """
        Initialize the data extractor.

        Args:
            h5file: Open h5py.File object
            format_type: 'Oxford' or 'EDAX'
        """
        self.h5file = h5file
        self.format = format_type
        self.root_key = self._find_root_key()
        # Caches
        self._grid_cache = None
        self._elements_cache = None
        self._electron_images_cache = None
        # Pattern cache for smooth navigation (Phase 7)
        self._pattern_cache = {}
        self._pattern_cache_max_size = 50  # Cache last 50 patterns

    def _find_root_key(self):
        """Find the acquisition slot (Oxford: '1', EDAX: sample name).

        Prefers a slot that holds EBSD data — unchanged behaviour for every
        normal file. Falls back to a slot that only has EDS or electron images:
        Aztec "Elementverteilungsdaten" acquisitions have no ``EBSD`` group at
        all, and returning None there made the extractor report no elements and
        no images, so the EDS viewer had nothing to show.
        """
        for key in self.h5file.keys():
            if isinstance(self.h5file[key], h5py.Group):
                if f'{key}/EBSD' in self.h5file:
                    return key
        for key in self.h5file.keys():
            if isinstance(self.h5file[key], h5py.Group):
                if f'{key}/EDS' in self.h5file or f'{key}/Electron Image' in self.h5file:
                    return key
        return None

    def _safe_read(self, group_path, key, default=None):
        """Safely read a value from HDF5"""
        try:
            full_path = f'{group_path}/{key}'
            if full_path in self.h5file:
                val = self.h5file[full_path][()]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                elif isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                return val
        except Exception:
            pass
        return default

    def get_grid_dimensions(self):
        """
        Get scan grid dimensions.

        Returns:
            Tuple (n_rows, n_cols)
        """
        if self._grid_cache is not None:
            return self._grid_cache

        if self.root_key is None:
            return (1, 1)

        header_path = f'{self.root_key}/EBSD/Header'

        if self.format == 'Oxford':
            n_cols = self._safe_read(header_path, 'X Cells', 1)
            n_rows = self._safe_read(header_path, 'Y Cells', 1)
        else:  # EDAX
            n_cols = self._safe_read(header_path, 'nColumns', 1)
            n_rows = self._safe_read(header_path, 'nRows', 1)

        # EDS-only acquisitions have no EBSD header at all. Their grid lives in
        # the EDS header — without this fall-back every element map came back
        # as a flat 1-D array on a 1x1 "grid" and could not be displayed.
        if int(n_cols) <= 1 and int(n_rows) <= 1:
            eds_header = f'{self.root_key}/EDS/Header'
            eds_cols = self._safe_read(eds_header, 'X Cells', 0)
            eds_rows = self._safe_read(eds_header, 'Y Cells', 0)
            if int(eds_cols) > 0 and int(eds_rows) > 0:
                n_cols, n_rows = eds_cols, eds_rows

        n_rows_i, n_cols_i = int(n_rows), int(n_cols)
        # Fail loudly on corrupt headers (e.g. X Cells = 0) rather than
        # silently clamping to a 1x1 grid — a silent clamp caused every
        # pattern lookup to map to (0, 0) and load wrong data without any
        # error visible to the user. Callers see a clear ValueError with
        # the actual reported values.
        if n_rows_i < 1 or n_cols_i < 1:
            raise ValueError(
                f"HDF5 header reports non-positive grid dimensions "
                f"(n_rows={n_rows_i}, n_cols={n_cols_i}) — file appears "
                f"corrupt or missing scan geometry."
            )
        self._grid_cache = (n_rows_i, n_cols_i)
        return self._grid_cache

    _AREA_GROUPS = {
        "ebsd": "EBSD",
        "eds": "EDS",
        "electron_image": "Electron Image",
    }

    def get_pixel_sizes(self):
        """Physical size of one pixel, PER ACQUISITION AREA, in microns.

        The areas do not share a scale. On a real file measured 2026-08-14:

            EBSD / EDS      12 x 9      X Step 0.6579 um
            Electron Image  1024 x 768  X Step 0.0621 um

        — a factor of 10.6 on a different field of view. Anything that draws a
        scale bar has to ask for the area the image came from; using one global
        step size silently mis-scales every electron-image export.

        Returns
        -------
        dict
            ``{"ebsd"|"eds"|"electron_image": {"x", "y", "units", "source"}}``,
            with ``None`` for an area that is absent or carries no geometry —
            never a guessed value, because a wrong scale bar is worse than no
            scale bar.
        """
        out = {}
        for key, group in self._AREA_GROUPS.items():
            out[key] = self._pixel_size_for(group)
        return out

    def _pixel_size_for(self, group_name):
        if self.root_key is None:
            return None
        header = f"{self.root_key}/{group_name}/Header"
        if header not in self.h5file:
            return None

        sx = self._safe_read(header, "X Step")
        sy = self._safe_read(header, "Y Step", sx)
        if sx is not None and float(sx) > 0:
            return {"x": float(sx), "y": float(sy if sy else sx),
                    "units": "um", "source": "step"}

        # Older exports: derive from the field of view and the cell count.
        bbox = self._safe_read(header, "Bounding Box Size")
        n_cols = self._safe_read(header, "X Cells")
        n_rows = self._safe_read(header, "Y Cells")
        try:
            if bbox is not None and n_cols and n_rows:
                bw, bh = float(bbox[0]), float(bbox[1])
                if bw > 0 and bh > 0 and int(n_cols) > 0 and int(n_rows) > 0:
                    return {"x": bw / int(n_cols), "y": bh / int(n_rows),
                            "units": "um", "source": "bounding_box"}
        except (TypeError, IndexError, ValueError):
            pass
        return None

    def get_pattern_count(self):
        """Get total number of patterns"""
        n_rows, n_cols = self.get_grid_dimensions()
        return n_rows * n_cols

    def get_pattern_shape(self, pattern_type='processed'):
        """
        Get shape of individual patterns.

        Returns:
            Tuple (height, width)
        """
        if self.root_key is None:
            return (128, 128)

        data_path = f'{self.root_key}/EBSD/Data'

        if self.format == 'Oxford':
            key = 'Unprocessed Patterns' if pattern_type == 'raw' else 'Processed Patterns'
            path = f'{data_path}/{key}'
        else:  # EDAX
            path = f'{data_path}/Pattern'

        if path in self.h5file:
            shape = self.h5file[path].shape
            return (shape[-2], shape[-1])

        return (128, 128)

    def get_pattern_at_index(self, index, pattern_type='processed'):
        """
        Get a single pattern by flat index (LAZY LOADING with CACHING).

        This loads only ONE pattern (~20KB) instead of all patterns (~200MB+).
        Caches recently accessed patterns for smooth slider navigation.

        Args:
            index: Flat index (0 to n_patterns-1)
            pattern_type: 'processed' or 'raw'

        Returns:
            2D numpy array (height, width) or None if not found
        """
        if self.root_key is None:
            return None

        # Check cache first
        cache_key = (index, pattern_type)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        data_path = f'{self.root_key}/EBSD/Data'

        if self.format == 'Oxford':
            key = 'Unprocessed Patterns' if pattern_type == 'raw' else 'Processed Patterns'
            path = f'{data_path}/{key}'
        else:  # EDAX
            path = f'{data_path}/Pattern'

        if path not in self.h5file:
            return None

        patterns_ds = self.h5file[path]

        # Check dimensions and extract single pattern.
        # Reject negative indices explicitly — numpy would happily wrap a
        # -1 into "last pattern" and silently return wrong data otherwise.
        if index < 0:
            return None
        pattern = None
        if patterns_ds.ndim == 3:
            # Shape: (n_points, height, width)
            if index >= patterns_ds.shape[0]:
                return None
            pattern = patterns_ds[index, :, :]
        elif patterns_ds.ndim == 4:
            # Shape: (n_rows, n_cols, height, width)
            n_rows, n_cols = self.get_grid_dimensions()
            row = index // n_cols
            col = index % n_cols
            if row < 0 or col < 0 or row >= n_rows or col >= n_cols:
                return None
            pattern = patterns_ds[row, col, :, :]

        # Add to cache
        if pattern is not None:
            self._add_to_cache(cache_key, pattern)

        return pattern

    def get_scalar_map(self, dataset_path: str) -> np.ndarray | None:
        """Read a 1D per-pixel dataset and reshape to (n_rows, n_cols).

        Returns None if the dataset is absent. Raises ValueError if shape
        mismatch — fail loud per project convention.
        """
        if dataset_path not in self.h5file:
            return None
        arr = self.h5file[dataset_path][:]
        n_rows, n_cols = self.get_grid_dimensions()
        expected = n_rows * n_cols
        if arr.ndim != 1 or arr.size != expected:
            raise ValueError(
                f"Dataset {dataset_path!r}: expected 1D array of length "
                f"{expected} (={n_rows}×{n_cols}), got shape={arr.shape}"
            )
        return arr.reshape(n_rows, n_cols)

    def get_phases_metadata(self) -> list[dict]:
        """Return list of phases with name/lattice/color/etc. Empty list if no phases."""
        if self.root_key is None:
            return []
        phases_path = f"{self.root_key}/EBSD/Header/Phases"
        if phases_path not in self.h5file:
            return []
        phases = []
        for key in sorted(self.h5file[phases_path].keys(), key=lambda k: int(k) if k.isdigit() else 0):
            g = self.h5file[f"{phases_path}/{key}"]
            def _read(name, default=None):
                if name not in g:
                    return default
                v = g[name][()]
                # H5OINA wraps everything as shape (1, N) or (1,) — squeeze the
                # leading singleton dim so Color becomes [r,g,b], Phase Name
                # becomes a bare bytes/str, Space Group becomes a bare int.
                if isinstance(v, np.ndarray):
                    v = np.squeeze(v)
                    if v.ndim == 0:
                        v = v.item()
                    elif v.ndim == 1:
                        v = v.tolist()
                    else:
                        v = v.tolist()
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="replace")
                return v
            phases.append({
                "id": int(key) if key.isdigit() else key,
                "name": _read("Phase Name", "Unknown"),
                "color": _read("Color", [128, 128, 128]),
                "lattice_dimensions": _read("Lattice Dimensions", [0, 0, 0]),
                "lattice_angles": _read("Lattice Angles", [90, 90, 90]),
                "space_group": _read("Space Group", 0),
                "laue_group": _read("Laue Group", 0),
                "n_reflectors": _read("Number Reflectors", 0),
                "reference": _read("Reference", ""),
            })
        return phases

    def get_phase_map(self) -> np.ndarray | None:
        """Return per-pixel phase ID array (n_rows, n_cols) or None."""
        if self.root_key is None:
            return None
        p = f"{self.root_key}/EBSD/Data/Phase"
        if p not in self.h5file:
            return None
        arr = self.h5file[p][:]
        n_rows, n_cols = self.get_grid_dimensions()
        return arr.reshape(n_rows, n_cols)

    def _read_frame_header_meta(self) -> dict:
        """Read frame-relevant header fields from the current file.

        Mirrors backend.api.services.orientation_ground_truth.load_vendor_orientations
        so apply_reference_frame_correction receives the same metadata regardless
        of code path. Returns an empty dict on any error — the caller is robust
        to that (Oxford/EDAX adapters under CS1 are identity and ignore header_meta).
        """
        try:
            if self.root_key is None:
                return {}
            header_path = f"{self.root_key}/EBSD/Header"
            header = self.h5file[header_path]
            fmt = str(self.format).lower()
            import numpy as _np
            if fmt.startswith("oxford"):
                soe = _np.asarray(header["Specimen Orientation Euler"][:],
                                  dtype=_np.float64).reshape(-1)[:3].tolist()
                return {
                    "specimen_orientation_euler": soe,
                    "scanning_rotation_angle":
                        float(_np.asarray(header["Scanning Rotation Angle"])[0]),
                    "tilt_angle": float(_np.asarray(header["Tilt Angle"])[0]),
                }
            if fmt.startswith(("edax", "tsl")):
                cs_id = None
                if "Coordinate System" in header and "ID" in header["Coordinate System"]:
                    cs_id = int(_np.asarray(header["Coordinate System"]["ID"])[0])
                return {"coordinate_system_id": cs_id}
        except Exception:
            return {}
        return {}

    def compute_ipf_map(self, direction: str = "Z") -> np.ndarray | None:
        """Compute per-pixel IPF RGB using orix. Returns (H, W, 3) uint8 or None.

        Mirrors the pattern in tools/phase_map_generator.compute_ipf_colors —
        reads Euler + Phase, builds Rotation, applies IPFColorKeyTSL per phase.
        """
        if self.root_key is None:
            return None
        euler_path = f"{self.root_key}/EBSD/Data/Euler"
        phase_path = f"{self.root_key}/EBSD/Data/Phase"
        if euler_path not in self.h5file or phase_path not in self.h5file:
            return None

        from orix.quaternion import Rotation
        from orix.vector import Vector3d
        from orix.plot import IPFColorKeyTSL
        from orix.crystal_map import Phase

        eulers = self.h5file[euler_path][:]  # (N, 3) radians per Aztec convention
        phase_ids = self.h5file[phase_path][:]
        n_rows, n_cols = self.get_grid_dimensions()
        n = n_rows * n_cols
        if eulers.shape != (n, 3):
            raise ValueError(f"Euler shape {eulers.shape} != ({n}, 3)")

        direction_map = {"X": Vector3d.xvector(), "Y": Vector3d.yvector(), "Z": Vector3d.zvector()}
        if direction not in direction_map:
            raise ValueError(f"Direction must be X|Y|Z, got {direction!r}")

        rotations = Rotation.from_euler(eulers, direction="lab2crystal")
        from backend.api.services.orientation_frame import apply_reference_frame_correction
        vendor = "oxford" if str(self.format).lower().startswith("oxford") else \
                 "edax" if str(self.format).lower().startswith(("edax", "tsl")) else \
                 "bruker"
        rotations = apply_reference_frame_correction(
            rotations,
            vendor=vendor,
            header_meta=self._read_frame_header_meta(),
            source="vendor",
        )
        rgb = np.zeros((n, 3), dtype=np.float32)

        phases_meta = self.get_phases_metadata()
        for pmeta in phases_meta:
            pid = int(pmeta["id"])
            mask = phase_ids == pid
            if not mask.any():
                continue
            # Derive the actual point group from the phase's space_group so
            # non-cubic phases (hex Mg 6/mmm, trigonal hematite, etc.) get
            # the correct IPF colors. Mirrors compute_ipf_colors() in
            # tools/phase_map_generator.py (lines 120-144): try space_group
            # via orix's Phase(space_group=int) constructor, fall back to
            # any explicit point_group, and fail loud if neither resolves.
            space_group = pmeta.get("space_group")
            point_group = pmeta.get("point_group")
            phase_name = pmeta.get("name", "?")
            try:
                if space_group and int(space_group) > 0:
                    phase = Phase(space_group=int(space_group))
                elif point_group:
                    phase = Phase(point_group=point_group)
                else:
                    raise ValueError(
                        f"Phase {pid} ({phase_name}) has no space_group "
                        f"and no point_group; cannot derive symmetry for "
                        f"IPF coloring"
                    )
                if phase.point_group is None:
                    raise ValueError(
                        f"Phase {pid} ({phase_name}, space_group="
                        f"{space_group}): orix returned Phase with no "
                        f"point_group — cannot color IPF"
                    )
            except Exception as e:
                logger.exception(
                    f"IPF coloring: failed to build orix Phase for phase "
                    f"{pid} ({phase_name}, space_group={space_group}, "
                    f"point_group={point_group}): {e}"
                )
                raise
            try:
                key = IPFColorKeyTSL(
                    phase.point_group, direction=direction_map[direction]
                )
                ipf = key.orientation2color(rotations[mask])
                rgb[mask] = ipf
            except Exception as e:
                logger.exception(
                    f"IPF coloring: orientation2color failed for phase "
                    f"{pid} ({phase_name}, point_group={phase.point_group}): {e}"
                )
                raise

        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(n_rows, n_cols, 3)
        return rgb_u8

    def get_eds_spectrum(self, index: int) -> dict | None:
        """Return spectrum + energy axis for one pixel, or None if no EDS in file."""
        if self.root_key is None:
            return None
        spec_path = f"{self.root_key}/EDS/Data/Spectrum"
        hdr_path = f"{self.root_key}/EDS/Header"
        if spec_path not in self.h5file:
            return None
        spec_ds = self.h5file[spec_path]
        if index < 0 or index >= spec_ds.shape[0]:
            raise ValueError(f"Spectrum index {index} out of range [0, {spec_ds.shape[0]})")
        counts = spec_ds[index, :].astype(np.int32)

        # Energy axis from header. Use np.squeeze to handle H5OINA's (1,) wrapping
        # of header scalars (same pattern as Task 2's get_phases_metadata).
        def _hdr(name, default):
            p = f"{hdr_path}/{name}"
            if p not in self.h5file:
                return default
            v = self.h5file[p][()]
            sq = np.squeeze(v)
            if sq.ndim == 0:
                return float(sq.item())
            return float(default)
        channel_width_eV = _hdr("Channel Width", 10.0)
        start_channel_eV = _hdr("Start Channel", 0.0)
        n_channels = int(_hdr("Number Channels", len(counts)))
        energy_eV = start_channel_eV + np.arange(n_channels) * channel_width_eV
        energy_keV = (energy_eV / 1000.0).astype(np.float32)

        def _per_pixel(ds_name):
            p = f"{self.root_key}/EDS/Data/{ds_name}"
            if p not in self.h5file:
                return None
            return float(self.h5file[p][index])

        return {
            "counts": counts.tolist(),
            "energy_axis_keV": energy_keV.tolist(),
            "live_time_s": _per_pixel("Live Time"),
            "real_time_s": _per_pixel("Real Time"),
            "channel_width_eV": channel_width_eV,
            "start_channel_eV": start_channel_eV,
            "number_channels": n_channels,
        }

    def get_eds_header(self) -> dict | None:
        """Return structured EDS header metadata, or None if no EDS in file."""
        if self.root_key is None:
            return None
        hp = f"{self.root_key}/EDS/Header"
        if hp not in self.h5file:
            return None

        def _read(name, default=None):
            p = f"{hp}/{name}"
            if p not in self.h5file:
                return default
            v = self.h5file[p][()]
            # H5OINA wraps everything in (1,) or (1,N); use np.squeeze + ndim dispatch
            # (same pattern as Task 2 had to invent for get_phases_metadata).
            # Do NOT use `v.size == 1` — H5OINA wraps scalars as (1,) and arrays
            # as (1, N), both of which size==1 would mishandle.
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            sq = np.squeeze(v)
            if sq.ndim == 0:
                val = sq.item()
                if isinstance(val, bytes):
                    return val.decode("utf-8", errors="replace")
                return val
            return sq.tolist()

        return {
            "channel_width_eV": _read("Channel Width"),
            "start_channel_eV": _read("Start Channel"),
            "number_channels": _read("Number Channels"),
            "energy_range_keV": _read("Energy Range"),
            "beam_voltage_kV": _read("Beam Voltage"),
            "detector": {
                "azimuth": _read("Detector Azimuth"),
                "elevation": _read("Detector Elevation"),
                "type_id": _read("Detector Type Id"),
                "serial": _read("Detector Serial Number"),
            },
            "window_type": _read("Window Type"),
            "process_time": _read("Process Time"),
            "drift_correction": _read("Drift Correction"),
            "strobe_area": _read("Strobe Area"),
            "strobe_fwhm_eV": _read("Strobe FWHM"),
            "magnification": _read("Magnification"),
            "working_distance_mm": _read("Working Distance"),
            "tilt_angle_deg": _read("Tilt Angle"),
            "acquisition_date": _read("Acquisition Date"),
        }

    def get_ebsd_header(self) -> dict | None:
        """Return structured EBSD header metadata, or None if no EBSD in file."""
        if self.root_key is None:
            return None
        hp = f"{self.root_key}/EBSD/Header"
        if hp not in self.h5file:
            return None

        def _read(name, default=None):
            p = f"{hp}/{name}"
            if p not in self.h5file:
                return default
            v = self.h5file[p][()]
            # H5OINA wraps everything in (1,) or (1,N); use np.squeeze + ndim dispatch
            # (same pattern as Task 2/4/5). Do NOT use `v.size == 1` — H5OINA wraps
            # scalars as (1,) and arrays as (1, N), both of which size==1 mishandles.
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            sq = np.squeeze(v)
            if sq.ndim == 0:
                val = sq.item()
                if isinstance(val, bytes):
                    return val.decode("utf-8", errors="replace")
                return val
            return sq.tolist()

        def _stage():
            sp = f"{hp}/Stage Position"
            if sp not in self.h5file:
                return {}
            out = {}
            for k in self.h5file[sp].keys():
                v = self.h5file[f"{sp}/{k}"][()]
                sq = np.squeeze(v)
                if sq.ndim == 0:
                    out[k.lower().replace(" ", "_")] = float(sq.item())
                else:
                    out[k.lower().replace(" ", "_")] = sq.tolist()
            return out

        return {
            "grid": {
                "x_cells": _read("X Cells"),
                "y_cells": _read("Y Cells"),
                "x_step_um": _read("X Step"),
                "y_step_um": _read("Y Step"),
                "scanning_rotation_angle_deg": _read("Scanning Rotation Angle"),
            },
            "pattern": {
                "height": _read("Pattern Height"),
                "width": _read("Pattern Width"),
                "acquired_height": _read("Acquired Pattern Height"),
                "acquired_width": _read("Acquired Pattern Width"),
                "acquisition_speed": _read("Acquisition Speed"),
                "frames_averaged": _read("Number Frames Averaged"),
            },
            "microscope": {
                "beam_voltage_kV": _read("Beam Voltage"),
                "working_distance_mm": _read("Working Distance"),
                "magnification": _read("Magnification"),
                "sample_tilt_deg": _read("Tilt Angle"),
                "tilt_axis_deg": _read("Tilt Axis"),
            },
            "detector": {
                "orientation_euler_deg": _read("Detector Orientation Euler"),
                "insertion_distance_mm": _read("Detector Insertion Distance"),
                "exposure_ms": _read("Camera Exposure Time"),
                "gain": _read("Camera Gain"),
                "mode": _read("Camera Mode"),
            },
            "band_detection": {
                "hough_resolution": _read("Hough Resolution"),
                "n_bands_detected": _read("Number Bands Detected"),
                "circle_center_x": _read("Band Detection Circle Center X"),
                "circle_center_y": _read("Band Detection Circle Center Y"),
                "circle_radius": _read("Band Detection Circle Radius"),
            },
            "background_correction": {
                "static": _read("Static Background Correction"),
                "auto": _read("Auto Background Correction"),
            },
            "stage": _stage(),
            "phases": self.get_phases_metadata(),
        }

    def get_static_background(self, kind: str = "processed") -> np.ndarray | None:
        """Return the static background pattern (processed or unprocessed) as a 2D array.

        Returns None if no file is open or the dataset is not present in this file.
        Raises ValueError on invalid `kind`.
        """
        if self.root_key is None:
            return None
        if kind == "processed":
            p = f"{self.root_key}/EBSD/Header/Processed Static Background"
        elif kind == "unprocessed":
            p = f"{self.root_key}/EBSD/Header/Unprocessed Static Background"
        else:
            raise ValueError(f"kind must be 'processed' or 'unprocessed', got {kind!r}")
        if p not in self.h5file:
            return None
        return self.h5file[p][:]

    def get_aztec_pixel(self, index: int) -> dict | None:
        """Return the full per-pixel Aztec record at flat `index`.

        Consolidates indexing data (phase, Euler, MAD, BC, BS, bands, quality,
        error, PC) and EDS timing (live/real time) into one record.

        Returns None if no file is open. Raises ValueError if index is out of
        range. Optional fields (e.g. EDS Live Time on EBSD-only files) become
        None rather than crashing.
        """
        if self.root_key is None:
            return None
        n_rows, n_cols = self.get_grid_dimensions()
        if not (0 <= index < n_rows * n_cols):
            raise ValueError(f"index {index} out of range")

        def _scalar(ds_name):
            p = f"{self.root_key}/EBSD/Data/{ds_name}"
            if p not in self.h5file:
                return None
            v = self.h5file[p][index]
            if isinstance(v, np.ndarray):
                return v.tolist()
            if hasattr(v, "item"):
                return v.item()
            return v

        def _eds_scalar(ds_name):
            p = f"{self.root_key}/EDS/Data/{ds_name}"
            if p not in self.h5file:
                return None
            return float(self.h5file[p][index])

        phase_id = _scalar("Phase")
        phases_meta = self.get_phases_metadata()
        phase_match = next(
            (p for p in phases_meta if int(p["id"]) == int(phase_id or -1)),
            None,
        )

        euler_rad = _scalar("Euler")  # Aztec stores radians
        euler_deg = (
            [float(np.degrees(e)) for e in euler_rad]
            if euler_rad is not None
            else None
        )

        pcx = _scalar("Pattern Center X")
        pcy = _scalar("Pattern Center Y")
        dd = _scalar("Detector Distance")

        return {
            "phase_id": int(phase_id) if phase_id is not None else None,
            "phase_name": phase_match["name"] if phase_match else None,
            "phase_color": phase_match["color"] if phase_match else None,
            "euler_deg": euler_deg,
            "mad_deg": _scalar("Mean Angular Deviation"),
            "band_contrast": _scalar("Band Contrast"),
            "band_slope": _scalar("Band Slope"),
            "bands": _scalar("Bands"),
            "pattern_quality": _scalar("Pattern Quality"),
            "error_code": _scalar("Error"),
            "pc": [pcx, pcy, dd] if pcx is not None else None,
            "live_time_s": _eds_scalar("Live Time"),
            "real_time_s": _eds_scalar("Real Time"),
            "beam_position_um": [_scalar("Beam Position X"), _scalar("Beam Position Y")],
            "scan_position_um": [_scalar("X"), _scalar("Y")],
        }

    def _add_to_cache(self, key, pattern):
        """Add pattern to cache with FIFO eviction."""
        # Evict oldest entries if cache is full
        while len(self._pattern_cache) >= self._pattern_cache_max_size:
            oldest_key = next(iter(self._pattern_cache))
            del self._pattern_cache[oldest_key]

        self._pattern_cache[key] = pattern

    def clear_pattern_cache(self):
        """Clear the pattern cache."""
        self._pattern_cache.clear()

    def index_to_position(self, index):
        """
        Convert flat index to grid position.

        Args:
            index: Flat index (0 to n_patterns-1)

        Returns:
            Tuple (row, col)
        """
        n_rows, n_cols = self.get_grid_dimensions()
        row = index // n_cols
        col = index % n_cols
        return (row, col)

    def position_to_index(self, row, col):
        """
        Convert grid position to flat index.

        Args:
            row: Row index
            col: Column index

        Returns:
            Flat index
        """
        n_rows, n_cols = self.get_grid_dimensions()
        return row * n_cols + col

    def get_available_elements(self):
        """
        Get list of available EDS elements.

        Oxford files can have two structures:
        1. Old: /EDS/Data/Window Integral/{Element}/Counts (Dataset)
        2. New: /EDS/Data/Window Integral/{Element Kα1} (Dataset directly)

        Returns:
            List of element names as found in the file
        """
        if self._elements_cache is not None:
            return self._elements_cache

        elements = []

        if self.format != 'Oxford' or self.root_key is None:
            self._elements_cache = elements
            return elements

        # Oxford EDS path
        eds_path = f'{self.root_key}/EDS/Data/Window Integral'

        if eds_path in self.h5file:
            eds_group = self.h5file[eds_path]
            for key in eds_group.keys():
                item = eds_group[key]
                # New structure: Element is directly a Dataset (e.g., "Fe Kα1")
                if isinstance(item, h5py.Dataset):
                    elements.append(key)
                # Old structure: Element is a Group with 'Counts' inside
                elif isinstance(item, h5py.Group):
                    if 'Counts' in item:
                        elements.append(key)

        self._elements_cache = sorted(elements)
        return self._elements_cache

    def get_element_map(self, element_name):
        """
        Get EDS element map data.

        Supports two structures:
        1. Old: /EDS/Data/Window Integral/{Element}/Counts
        2. New: /EDS/Data/Window Integral/{Element Kα1} (Dataset directly)

        Args:
            element_name: Element name as returned by get_available_elements()

        Returns:
            1D numpy array (n_points,) - use reshape with get_grid_dimensions()
        """
        if self.format != 'Oxford' or self.root_key is None:
            return None

        base_path = f'{self.root_key}/EDS/Data/Window Integral/{element_name}'

        # Try new structure first: element is directly a dataset
        if base_path in self.h5file:
            item = self.h5file[base_path]
            if isinstance(item, h5py.Dataset):
                return item[:]

        # Try old structure: element is a group with Counts inside
        counts_path = f'{base_path}/Counts'
        if counts_path in self.h5file:
            return self.h5file[counts_path][:]

        logger.warning("EDS element map not found: %s", element_name)
        return None

    def get_element_map_2d(self, element_name):
        """
        Get EDS element map reshaped to 2D grid.

        Args:
            element_name: Element symbol (e.g., 'Fe', 'Ni')

        Returns:
            2D numpy array (n_rows, n_cols) or None
        """
        data = self.get_element_map(element_name)
        if data is None:
            return None

        n_rows, n_cols = self.get_grid_dimensions()
        if data.size == n_rows * n_cols:
            return data.reshape(n_rows, n_cols)
        return data

    def get_band_contrast_map(self):
        """
        Return Band Contrast as a 2D float32 array (n_rows, n_cols), or None
        if not present.

        H5OINA stores Band Contrast as a flat 1D array of length n_rows*n_cols
        under <root_key>/EBSD/Data/Band Contrast. We reshape to 2D using the
        grid dims from the header. This is a SINGLE dataset read and avoids
        the per-pattern Python loop that the minimap fallback used to do.

        Returns:
            2D numpy array (n_rows, n_cols) of dtype float32, or None when
            the file has no Band Contrast dataset (e.g. EDAX, synthetic data).

        Raises:
            ValueError: If the Band Contrast dataset exists but its shape
                does not match the header-reported grid (size mismatch on
                a 1D array, or unexpected ndim). Per the project's
                fail-loud-not-silent contract: a header/data mismatch
                must surface, not silently fall through.
        """
        if self.format != 'Oxford' or self.root_key is None:
            return None

        path = f"{self.root_key}/EBSD/Data/Band Contrast"
        if path not in self.h5file:
            return None

        n_rows, n_cols = self.get_grid_dimensions()
        arr = self.h5file[path][:]
        if arr.ndim == 1:
            if arr.size != n_rows * n_cols:
                raise ValueError(
                    f"Band Contrast at {path} has size {arr.size}, expected "
                    f"{n_rows * n_cols} ({n_rows}x{n_cols}). Header/data mismatch."
                )
            arr = arr.reshape(n_rows, n_cols)
        elif arr.ndim != 2:
            raise ValueError(
                f"Band Contrast at {path} has unexpected ndim={arr.ndim}, "
                f"shape={arr.shape}. Expected 1D or 2D."
            )
        return arr.astype(np.float32)

    def get_available_electron_images(self):
        """
        Get list of available electron images (SE, BSE, FSE).

        Searches recursively for datasets under /Electron Image/Data/ only.
        Oxford H5OINA files store electron-detector images as:
        - /1/Electron Image/Data/SE/Elektronenbild 1    (real secondary electron)
        - /1/Electron Image/Data/FSE/Oben links         (forward-scatter detector)
        - /1/Electron Image/Data/FSE/Oben rechts        (etc., 4-5 quadrants)
        - /1/Electron Image/Data/BSE/...                (backscatter, when present)

        These are RAW detector images, useful as scan-grid overlays.

        EXPLICITLY EXCLUDED — this method does NOT return Aztec-pre-indexed
        images stored under /1/Layered Image/ (typically
        /Layered Image/EBSD-Schichtbild 1/Data/Color and
        /Layered Image/EDS-Schichtbild 1/Data/Color). Those are
        already-indexed RGB outputs produced inside Aztec; we produce
        our own indexing in this app and do not want Aztec's pre-renders
        polluting the layer list.

        Returns:
            List of image paths relative to Data/ folder
            e.g., ['FSE/Oben links', 'SE/Elektronenbild 1', ...]
        """
        if self._electron_images_cache is not None:
            return self._electron_images_cache

        images = []

        if self.format != 'Oxford' or self.root_key is None:
            self._electron_images_cache = images
            return images

        # Oxford electron image path
        ei_data_path = f'{self.root_key}/Electron Image/Data'

        if ei_data_path in self.h5file:
            data_group = self.h5file[ei_data_path]

            # Recursively find all datasets
            def find_datasets(group, prefix=''):
                for key in group.keys():
                    item = group[key]
                    full_path = f"{prefix}/{key}" if prefix else key
                    if isinstance(item, h5py.Dataset):
                        images.append(full_path)
                    elif isinstance(item, h5py.Group):
                        find_datasets(item, full_path)

            find_datasets(data_group)

        self._electron_images_cache = images
        return self._electron_images_cache

    def get_electron_image(self, image_name):
        """
        Get electron image data (SE or BSE).

        Handles various shapes:
        - 1D (n_points,) -> reshapes to (n_rows, n_cols)
        - 2D (height, width) -> returns as-is
        - 3D (slices, height, width) -> returns middle slice

        Args:
            image_name: Image path relative to Data/ folder
                       e.g., 'SE/Elektronenbild 1' or 'FSE/Oben links'

        Returns:
            2D numpy array
        """
        if self.format != 'Oxford' or self.root_key is None:
            return None

        # image_name can now be a path like "SE/Elektronenbild 1"
        path = f'{self.root_key}/Electron Image/Data/{image_name}'

        if path not in self.h5file:
            logger.warning("Electron image path not found: %s", path)
            return None

        data = self.h5file[path][:]

        # Handle different shapes
        if data.ndim == 1:
            n_rows, n_cols = self.get_grid_dimensions()
            if data.size == n_rows * n_cols:
                return data.reshape(n_rows, n_cols)
            # Try to find reasonable dimensions
            total = data.size
            for rows in range(int(np.sqrt(total)), 0, -1):
                if total % rows == 0:
                    cols = total // rows
                    return data.reshape(rows, cols)
            return data
        elif data.ndim == 2:
            return data
        elif data.ndim == 3:
            return data[len(data) // 2]

        return data

    def detect_available_features(self):
        """
        Detect what features are available in the file.

        Returns:
            dict with keys:
                - has_patterns: bool
                - has_raw_patterns: bool
                - has_eds: bool
                - has_electron_images: bool
                - pattern_count: int
                - pattern_shape: (height, width)
                - grid_shape: (n_rows, n_cols)
                - eds_elements: list of element names
                - electron_images: list of image names
        """
        features = {
            'has_patterns': False,
            'has_raw_patterns': False,
            'has_eds': False,
            'has_electron_images': False,
            'pattern_count': 0,
            'pattern_shape': (0, 0),
            'grid_shape': (0, 0),
            'eds_elements': [],
            'electron_images': [],
        }

        if self.root_key is None:
            return features

        data_path = f'{self.root_key}/EBSD/Data'

        # Check patterns
        if self.format == 'Oxford':
            features['has_patterns'] = f'{data_path}/Processed Patterns' in self.h5file
            features['has_raw_patterns'] = f'{data_path}/Unprocessed Patterns' in self.h5file
        else:  # EDAX
            features['has_patterns'] = f'{data_path}/Pattern' in self.h5file

        # Get counts and shapes
        features['grid_shape'] = self.get_grid_dimensions()
        features['pattern_count'] = self.get_pattern_count()
        features['pattern_shape'] = self.get_pattern_shape()

        # Check EDS
        features['eds_elements'] = self.get_available_elements()
        features['has_eds'] = len(features['eds_elements']) > 0

        # Check electron images
        features['electron_images'] = self.get_available_electron_images()
        features['has_electron_images'] = len(features['electron_images']) > 0

        return features

    def enumerate_layers(self) -> dict:
        """Walk known dataset paths, return only what's present in this file.

        Returns a catalog grouped by category (quality, indexing, geometry,
        eds, electron). Each entry has at minimum: id, name, path, kind.
        Used by the cockpit-style frontend to render available map layers
        without having to probe each endpoint individually.
        """
        if self.root_key is None:
            return {"quality": [], "indexing": [], "geometry": [], "eds": [], "electron": []}

        rk = self.root_key
        quality_specs = [
            ("Band Contrast", "Band Contrast"),
            ("Band Slope", "Band Slope"),
            ("Bands", "Bands"),
            ("Mean Angular Deviation", "Mean Angular Deviation"),
            ("Pattern Quality", "Pattern Quality"),
            ("Error", "Error"),
        ]
        geometry_specs = [
            ("Pattern Center X", "Pattern Center X"),
            ("Pattern Center Y", "Pattern Center Y"),
            ("Detector Distance", "Detector Distance"),
            ("Beam Position X", "Beam Position X"),
            ("Beam Position Y", "Beam Position Y"),
        ]
        eds_scalar_specs = [
            ("Live Time", "Live Time"),
            ("Real Time", "Real Time"),
        ]

        def _scalar_layer(category, label, ds_name, group="EBSD"):
            path = f"/{rk}/{group}/Data/{ds_name}"
            if path not in self.h5file:
                return None
            ds = self.h5file[path]
            return {
                "id": f"{group.lower()}:{ds_name}",
                "name": label,
                "path": path,
                "kind": "scalar",
                "dtype": str(ds.dtype),
            }

        layers = {"quality": [], "indexing": [], "geometry": [], "eds": [], "electron": []}
        for label, name in quality_specs:
            layer = _scalar_layer("quality", label, name, "EBSD")
            if layer:
                layers["quality"].append(layer)
        for label, name in geometry_specs:
            layer = _scalar_layer("geometry", label, name, "EBSD")
            if layer:
                layers["geometry"].append(layer)
        for label, name in eds_scalar_specs:
            layer = _scalar_layer("eds_scalar", label, name, "EDS")
            if layer:
                layers["eds"].append(layer)

        # Indexing layers: phase + IPF (only if Euler+Phase exist)
        if f"/{rk}/EBSD/Data/Phase" in self.h5file:
            layers["indexing"].append({
                "id": "indexing:Phase",
                "name": "Phase",
                "path": f"/{rk}/EBSD/Data/Phase",
                "kind": "phase",
            })
        if f"/{rk}/EBSD/Data/Euler" in self.h5file and f"/{rk}/EBSD/Data/Phase" in self.h5file:
            for d in ("X", "Y", "Z"):
                layers["indexing"].append({
                    "id": f"indexing:IPF-{d}",
                    "name": f"IPF ({d})",
                    "path": f"/{rk}/EBSD/Data/Euler",
                    "kind": "ipf",
                    "direction": d,
                })

        # EDS element layers
        win_path = f"/{rk}/EDS/Data/Window Integral"
        if win_path in self.h5file:
            for el in sorted(self.h5file[win_path].keys()):
                layers["eds"].append({
                    "id": f"eds_element:{el}",
                    "name": el,
                    "path": f"{win_path}/{el}",
                    "kind": "eds_element",
                })

        # Electron images. We omit the `path` field because the frontend
        # dispatches by `kind` and never reads it; including a path here was
        # previously misleading (it pointed at /EBSD/Data/, while the actual
        # image lives under /Electron Image/Data/).
        e_images = self.get_available_electron_images()
        for name in e_images:
            layers["electron"].append({
                "id": f"electron:{name}",
                "name": name,
                "kind": "electron",
            })

        return layers
