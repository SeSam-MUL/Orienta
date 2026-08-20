"""CalibrationStore — Single source of truth for PC/detector per dataset.

Every loaded EBSD dataset gets a CalibrationEntry that stores its detector
geometry and pattern center (PC).  Derived datasets (deepcopy, frame-averaged)
inherit their parent's calibration.  PC Refinement updates the store, and
Indexing reads from it — no more fragile fallback chains.

Usage::

    from backend.api.services.calibration_store import calibration_store

    # On load
    calibration_store.register("Sample_B.h5oina", signal)

    # On deepcopy
    calibration_store.register_derived("Sample_B_copy", parent_name="Sample_B.h5oina")

    # After PC refinement
    calibration_store.update_pc("Sample_B.h5oina", [0.512, 0.331, 0.850])

    # For indexing
    detector = calibration_store.get_detector("Sample_B_copy")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationEntry:
    """Calibration data for a single EBSD dataset."""

    dataset_name: str
    detector_shape: Tuple[int, int]  # (nrows, ncols) of detector
    pc_single: np.ndarray  # shape (3,) — mean or refined PC
    pc_map: Optional[np.ndarray] = None  # shape (nrows, ncols, 3) per-pixel PC
    pc_source: str = "header"  # "header" | "refined" | "inherited" | "manual" | "propagated"
    sample_tilt: float = 70.0
    tilt: float = 0.0
    azimuthal: float = 0.0
    parent_name: Optional[str] = None

    def to_dict(self) -> dict:
        """JSON-serialisable summary for API responses.

        ``detector_shape`` may be a tuple of numpy integer types when populated
        from a kikuchipy EBSDDetector (whose ``.shape`` comes from numpy
        ``.shape`` indexing), so we coerce to plain ``int`` — FastAPI's
        ``jsonable_encoder`` does not handle ``numpy.int32`` and would otherwise
        return a 500 from any calibration GET endpoint.
        """
        return {
            "dataset_name": self.dataset_name,
            "detector_shape": [int(v) for v in self.detector_shape],
            "pc": [float(v) for v in self.pc_single],
            "pc_source": self.pc_source,
            "has_pc_map": self.pc_map is not None,
            "sample_tilt": float(self.sample_tilt),
            "tilt": float(self.tilt),
            "azimuthal": float(self.azimuthal),
            "parent_name": self.parent_name,
        }


class CalibrationStore:
    """Thread-safe registry of per-dataset calibration data."""

    def __init__(self) -> None:
        self._entries: Dict[str, CalibrationEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, signal) -> Optional[CalibrationEntry]:
        """Extract detector/PC from a kikuchipy EBSD signal and store it.

        Parameters
        ----------
        name : str
            Dataset name (used as key).
        signal : kikuchipy.signals.EBSD
            Loaded EBSD signal with a ``.detector`` attribute.

        Returns
        -------
        CalibrationEntry or None if no detector found on signal.
        """
        det = getattr(signal, "detector", None)
        if det is None:
            logger.warning("CalibrationStore.register(%r): signal has no detector", name)
            return None

        pc_source = "header"
        try:
            pc_raw = np.array(det.pc)
            # kikuchipy always returns at least (1, 3) for a single PC.
            # Per-pixel PC has shape (nrows, ncols, 3) i.e. ndim==3,
            # or (N, 3) with N > 1 for a flat list of multiple PCs.
            is_per_pixel = (pc_raw.ndim == 3) or (pc_raw.ndim == 2 and pc_raw.shape[0] > 1)
            if is_per_pixel:
                pc_map = pc_raw.copy()
                pc_single = pc_raw.reshape(-1, 3).mean(axis=0)
                logger.info(
                    "register(%r): per-pixel PC shape %s, mean PC [%.4f, %.4f, %.4f]",
                    name, pc_raw.shape, *pc_single,
                )
            else:
                pc_single = pc_raw.flatten()[:3].copy()
                pc_map = None
                logger.info("register(%r): PC [%.4f, %.4f, %.4f]", name, *pc_single)
        except Exception:
            # Do NOT label a fabricated PC as "header" — that makes a guessed
            # (0.5,0.5,0.5) indistinguishable from a real header PC, and the
            # PC is the quantity indexing stands and falls on. Mark it
            # "missing" and log loudly so consumers / the UI can refuse to
            # index until a real PC is supplied.
            logger.error(
                "register(%r): could not read PC from detector — storing placeholder "
                "(0.5,0.5,0.5) with pc_source='missing'. Calibrate/refine the PC before indexing.",
                name, exc_info=True,
            )
            pc_single = np.array([0.5, 0.5, 0.5])
            pc_map = None
            pc_source = "missing"

        entry = CalibrationEntry(
            dataset_name=name,
            detector_shape=tuple(det.shape),
            pc_single=pc_single,
            pc_map=pc_map,
            pc_source=pc_source,
            sample_tilt=float(getattr(det, "sample_tilt", 70.0)),
            tilt=float(getattr(det, "tilt", 0.0)),
            azimuthal=float(getattr(det, "azimuthal", 0.0)),
        )

        with self._lock:
            self._entries[name] = entry
        return entry

    def register_derived(self, name: str, parent_name: str) -> Optional[CalibrationEntry]:
        """Register a derived dataset (deepcopy) that inherits parent calibration.

        Returns
        -------
        CalibrationEntry or None if parent not found.
        """
        with self._lock:
            parent = self._entries.get(parent_name)
            if parent is None:
                logger.warning(
                    "register_derived(%r): parent %r not in store", name, parent_name
                )
                return None

            entry = CalibrationEntry(
                dataset_name=name,
                detector_shape=parent.detector_shape,
                pc_single=parent.pc_single.copy(),
                pc_map=parent.pc_map.copy() if parent.pc_map is not None else None,
                pc_source="inherited",
                sample_tilt=parent.sample_tilt,
                tilt=parent.tilt,
                azimuthal=parent.azimuthal,
                parent_name=parent_name,
            )
            self._entries[name] = entry

        logger.info(
            "register_derived(%r) from %r: PC [%.4f, %.4f, %.4f] (inherited)",
            name, parent_name, *entry.pc_single,
        )
        return entry

    def register_cropped(
        self, name: str, parent_name: str, window
    ) -> Optional[CalibrationEntry]:
        """Register a CROPPED dataset: same geometry, per-pixel PC map cut.

        ``register_derived`` inherits ``pc_map`` verbatim, which is right for a
        deepcopy and wrong for a crop — the map is indexed by navigation
        position, so on a cropped grid every entry would be off by the crop
        offset. ``window`` is a ``crop_window.CropWindow``.
        """
        with self._lock:
            parent = self._entries.get(parent_name)
            if parent is None:
                logger.warning(
                    "register_cropped(%r): parent %r not in store", name, parent_name
                )
                return None

            pc_map = None
            if parent.pc_map is not None:
                pc_map = np.array(window.apply(parent.pc_map), copy=True)

            entry = CalibrationEntry(
                dataset_name=name,
                detector_shape=parent.detector_shape,
                pc_single=parent.pc_single.copy(),
                pc_map=pc_map,
                pc_source="inherited",
                sample_tilt=parent.sample_tilt,
                tilt=parent.tilt,
                azimuthal=parent.azimuthal,
                parent_name=parent_name,
            )
            self._entries[name] = entry

        logger.info(
            "register_cropped(%r) from %r: window rows %d-%d cols %d-%d, "
            "pc_map %s",
            name, parent_name, window.row0, window.row0 + window.rows,
            window.col0, window.col0 + window.cols,
            "cut" if pc_map is not None else "absent",
        )
        return entry

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_pc(
        self, name: str, pc, source: str = "refined"
    ) -> bool:
        """Update the single-PC value for a dataset.

        Parameters
        ----------
        pc : array-like, shape (3,)
        source : str
            Origin of this PC value ("refined", "manual", etc.).

        Returns True if the entry existed and was updated.
        """
        pc_arr = np.asarray(pc, dtype=float).flatten()[:3]
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                logger.warning("update_pc(%r): not in store", name)
                return False
            prev_source = entry.pc_source
            had_map = entry.pc_map is not None

            if had_map and prev_source == "refined_map":
                # A genuine per-pixel refinement (update_pc_map: Aztec seed / grid
                # calibration) OUTRANKS a global single-PC update. Keep the map AND
                # its "refined_map" provenance so get_detector still serves the
                # per-pixel field and a later refine doesn't mistake it for a
                # shiftable header map. Record the requested single PC for get_pc
                # consumers only.
                entry.pc_single = pc_arr
                action = "kept-refined_map"
            elif had_map:
                # A header/inherited per-pixel PC (e.g. Oxford's smooth PC plane).
                # A global refine corrects the OFFSET — rigid-shift the whole map so
                # its mean lands on the refined PC while the spatial structure
                # (per-pixel deltas = the PC heat-map data) is preserved. get_detector
                # keeps serving a per-pixel map, so Hough region indexing stays intact,
                # and every mean-consumer now sees the refined PC. (For a constant map
                # this reduces to setting every pixel to pc_arr, i.e. equivalent to a
                # plain single-PC update — no special case needed.)
                # Note: a deepcopy inherits its parent's map tagged "inherited"
                # (not "refined_map"), so it is shifted here BY DESIGN — the shift
                # preserves per-pixel structure, and the parent's OWN genuine
                # refined_map is still spared by the branch above (the F2 propagation
                # writes the parent separately).
                delta = pc_arr - entry.pc_map.reshape(-1, 3).mean(axis=0)
                entry.pc_map = entry.pc_map + delta
                entry.pc_single = pc_arr
                entry.pc_source = source
                action = "rigid-shift"
            else:
                # No map: plain single-PC update (original behaviour).
                entry.pc_single = pc_arr
                entry.pc_source = source
                action = "single"
        logger.info("update_pc(%r): [%.4f, %.4f, %.4f] source=%s (%s)",
                    name, *pc_arr, source, action)
        return True

    def update_pc_map(self, name: str, pc_map: np.ndarray) -> bool:
        """Set per-pixel PC map for a dataset.

        Also updates pc_single to the mean of the map.
        """
        pc_map = np.asarray(pc_map, dtype=float)
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                logger.warning("update_pc_map(%r): not in store", name)
                return False
            entry.pc_map = pc_map.copy()
            entry.pc_single = pc_map.reshape(-1, 3).mean(axis=0)
            entry.pc_source = "refined_map"
        logger.info(
            "update_pc_map(%r): shape %s, mean [%.4f, %.4f, %.4f]",
            name, pc_map.shape, *entry.pc_single,
        )
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_detector(self, name: str):
        """Build an EBSDDetector with the best available PC for a dataset.

        Returns None if the dataset is not in the store.
        Uses pc_map if available, otherwise pc_single.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return None
            # Snapshot values under lock
            shape = entry.detector_shape
            pc = entry.pc_map.copy() if entry.pc_map is not None else entry.pc_single.copy()
            sample_tilt = entry.sample_tilt
            tilt = entry.tilt
            azimuthal = entry.azimuthal

        # Build detector outside lock (import may be slow first time)
        from kikuchipy.detectors import EBSDDetector

        det = EBSDDetector(
            shape=shape,
            pc=pc,
            sample_tilt=sample_tilt,
            tilt=tilt,
            azimuthal=azimuthal,
        )
        return det

    def get_pc(self, name: str) -> Optional[np.ndarray]:
        """Get single PC (mean if per-pixel available). Returns shape (3,) or None.

        Returns None when the stored PC is a fabricated placeholder
        (``pc_source == "missing"``, set by register() when the detector PC
        could not be read). This makes every consumer — /refine, batch PC
        inheritance, etc. — fail loud rather than index/refine against a
        guessed (0.5,0.5,0.5). A real PC (header/refined/...) is returned
        unchanged.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None or entry.pc_source == "missing":
                return None
            return entry.pc_single.copy()

    def get_entry(self, name: str) -> Optional[CalibrationEntry]:
        """Get the full CalibrationEntry (or None)."""
        with self._lock:
            return self._entries.get(name)

    def has_pc_map(self, name: str) -> bool:
        with self._lock:
            entry = self._entries.get(name)
            return entry is not None and entry.pc_map is not None

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def remove(self, name: str) -> bool:
        with self._lock:
            removed = self._entries.pop(name, None)
        if removed:
            logger.info("remove(%r)", name)
        return removed is not None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        logger.info("clear(): all entries removed")

    def get_all(self) -> Dict[str, CalibrationEntry]:
        """Return a shallow copy of all entries."""
        with self._lock:
            return dict(self._entries)

    def get_all_entries(self) -> Dict[str, CalibrationEntry]:
        """Alias for get_all() — return a shallow copy of all entries."""
        return self.get_all()

    def snapshot(self, names) -> Dict[str, CalibrationEntry]:
        """Deep-copy the entries for ``names`` for a per-file stash.

        The viewer stashes the outgoing file's calibration before
        ``clear()`` so derived-dataset PC/detector geometry survives a file
        round-trip (A -> B -> A). Deep-copies so later mutation of the live
        store (or the stash) can't bleed across.
        """
        import copy
        with self._lock:
            return {
                n: copy.deepcopy(self._entries[n])
                for n in names if n in self._entries
            }

    def restore(self, entries: Dict[str, CalibrationEntry], skip: Optional[str] = None) -> None:
        """Re-insert entries produced by :meth:`snapshot` (per-file restore).

        ``skip`` (the freshly re-registered raw dataset) is left untouched so
        a fresh header/refined PC is not overwritten by the stashed copy.
        Entries are deep-copied on insert so the live store never aliases the
        (now-consumed) stash.
        """
        import copy
        with self._lock:
            for name, entry in entries.items():
                if name == skip:
                    continue
                self._entries[name] = copy.deepcopy(entry)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._entries


# Module-level singleton — import this everywhere
calibration_store = CalibrationStore()
