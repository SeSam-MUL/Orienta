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
    #: Set when the PC (or PC map) of this entry came out of PC refinement, which
    #: runs PyEBSDIndex's PC optimiser: ``{"n_patterns": int, "method": "PSO"|"NM"}``.
    #: ``None`` for every other origin (file, typed, imported map). Read when a run
    #: is indexed, so its citation trail credits the software that produced the PC.
    #: Every write of the PC replaces it, so it can only describe the PC that is
    #: stored now.
    pc_refinement: Optional[dict] = None
    #: The PC the FILE itself gave when the dataset was registered (mean of its map),
    #: or ``None`` when it carried none (``pc_source == "missing"``) or the entry is a
    #: copy. Kept so that, when a calibration the user made is restored on a later
    #: load, it can be told whether the file has changed since: a file that now
    #: carries a PC (an .osc put beside it) or another one takes precedence.
    file_pc: Optional[np.ndarray] = None
    #: Canonical path of the file this entry was registered for. The store is keyed by
    #: the dataset NAME (the file stem), so two files with one stem share a key; the path
    #: says whose entry it is, and a guard asking about another file does not take it
    #: for that file's. ``None`` for an entry registered without one (it matches any).
    file_key: Optional[str] = None
    #: True for an entry that is a VIEW of its parent's file: a deepcopy or a crop
    #: (``register_derived`` / ``register_cropped``), which inherits the parent's
    #: ``file_key``. Such an entry is never "the entry of its file" (``name_of_file``).
    #: ``parent_name`` alone cannot say this: ``copy-pc`` also sets it, on the entry of
    #: the TARGET FILE, to record whose PC it took (and so that a later refine passes on
    #: along the family) - that entry stays the entry of its own file.
    is_view_of_parent: bool = False
    #: The acquisition settings of the FILE this entry was registered for (pattern shape, binning,
    #: camera and sample tilt, working distance; ``pc_calibrations.features_of_signal``), or ``None``
    #: for a copy and for an entry whose signal could not be read for them. A setting the file's
    #: format does not carry is ``None`` inside it (``pc_calibrations.settings_stated``): kikuchipy's
    #: detector defaults are not measurements. A calibration is compared with these before it is applied.
    features: Optional[dict] = None
    #: Where the FILE's PC came from (``pc_origin_of``). Unlike ``pc_source`` it does not change
    #: when the user types or refines a PC: it says what the file itself offered. ``None`` for an
    #: entry registered by older code.
    pc_origin: Optional[str] = None
    #: The convention the FILE's vendor states a PC in ("bruker" | "tsl" | "oxford", ``pc_calibrations.convention_of``)
    #: and whether the format says so; ``None`` / False for an entry registered by older code. Every PC in the
    #: store is in Bruker's convention whatever this says: it is what the page shows next to the numbers.
    convention: Optional[str] = None
    convention_known: bool = False
    #: True on the entry of a stash that dropped the per-pixel map the file's header brought (the file is read again
    #: when it is opened, ``CalibrationStore.without_header_maps``): ``pc_map`` is None, the file still has a map.
    map_dropped: bool = False

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
            "pc_origin": self.pc_origin,
            "has_pc_map": self.pc_map is not None,
            "sample_tilt": float(self.sample_tilt),
            "tilt": float(self.tilt),
            "azimuthal": float(self.azimuthal),
            "parent_name": self.parent_name,
        }


def _copy_refinement(refinement: Optional[dict]) -> Optional[dict]:
    return dict(refinement) if refinement else None


def _loader_pc_source(signal) -> Optional[str]:
    """What the loader recorded about where the signal's PC came from
    (``Signal.pc_source``: ``"osc"`` or ``"default"``); None when it recorded
    nothing, as for every format whose reader carries its own PC."""
    try:
        value = signal.metadata.get_item("Signal.pc_source", None)
    except Exception:  # noqa: BLE001 - a stand-in signal without metadata
        return None
    return str(value) if isinstance(value, str) else None


def _signal_features(signal) -> Optional[dict]:
    """The settings of a loaded file for ``CalibrationEntry.features``; ``None`` (and a log line) when
    the signal cannot give them: a calibration is then never applied to this entry without being told
    it could not be checked."""
    try:
        from backend.api.services import pc_calibrations
        return pc_calibrations.features_of_signal(signal)
    except Exception:  # noqa: BLE001 - a stand-in signal in a test, an odd reader
        logger.warning("could not read the acquisition settings of a loaded file", exc_info=True)
        return None


def _signal_convention(signal, file_key):
    """``(convention, known)`` of a loaded file (``pc_calibrations.convention_of``); ``(None, False)`` when the
    signal cannot say (a stand-in signal in a test, an odd reader)."""
    try:
        from backend.api.services import pc_calibrations
        return pc_calibrations.convention_of(signal, file_key)
    except Exception:  # noqa: BLE001
        logger.warning("could not tell the PC convention of a loaded file", exc_info=True)
        return None, False


def pc_origin_of(signal) -> str:
    """Where the file's own pattern centre came from, in one vocabulary:
    ``"apex"`` (an EDAX APEX metadata file), ``"osc"`` (an .osc sidecar), ``"orienta_export"``
    (the /Detector/pc of an Orienta export whose header has none), ``"header"`` (the manufacturer's
    header, the case where the loader recorded nothing) or ``"missing"`` (nothing supplied a PC;
    kikuchipy's placeholder is in use - the loader's tag for that is ``"default"``)."""
    tag = _loader_pc_source(signal)
    if tag == "default":
        return "missing"
    return tag or "header"


def _inherited_source(parent_source: str) -> str:
    """A copy inherits its parent's PC; a PC nobody supplied stays one nobody supplied."""
    return "missing" if parent_source == "missing" else "inherited"


class CalibrationStore:
    """Thread-safe registry of per-dataset calibration data."""

    def __init__(self) -> None:
        self._entries: Dict[str, CalibrationEntry] = {}
        self._lock = threading.Lock()
        #: ``None`` for a store of its own. Set by :meth:`over`: names the file whose entries these are, so that
        #: what is remembered per dataset name elsewhere (the step back, the calibration a file was given) is kept
        #: apart from an open file that has the same name.
        self.scope: Optional[str] = None

    @classmethod
    def over(cls, entries: Dict[str, CalibrationEntry], scope: str) -> "CalibrationStore":
        """A store whose entries ARE ``entries`` (the dict is shared, not copied): a write through it is a write to
        that dict. It is how the calibration of a file that is not open (the ``calibration`` of its stash in the
        viewer) is read and written by the code that works on a store. ``scope`` is the file's canonical path."""
        store = cls()
        store._entries = entries
        store.scope = scope
        return store

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, signal, file_key: Optional[str] = None) -> Optional[CalibrationEntry]:
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

        # The loader says when the reader had no PC and kikuchipy attached its
        # placeholder (an EDAX UP1/UP2 without an .osc): reading that PC works, but it
        # is not a measurement. Mark it "missing" like an unreadable one, so the
        # guards that refuse to work from a PC nobody supplied can fire.
        if pc_source == "header" and _loader_pc_source(signal) == "default":
            logger.warning(
                "register(%r): the file carries no pattern centre; kikuchipy's "
                "placeholder (%.2f, %.2f, %.2f) is stored with pc_source='missing'. "
                "Set or refine the PC before indexing.", name, *pc_single)
            pc_source = "missing"

        convention, convention_known = _signal_convention(signal, file_key)
        entry = CalibrationEntry(
            dataset_name=name,
            detector_shape=tuple(det.shape),
            pc_single=pc_single,
            pc_map=pc_map,
            pc_source=pc_source,
            sample_tilt=float(getattr(det, "sample_tilt", 70.0)),
            tilt=float(getattr(det, "tilt", 0.0)),
            azimuthal=float(getattr(det, "azimuthal", 0.0)),
            file_pc=None if pc_source == "missing" else np.asarray(pc_single, dtype=float).copy(),
            features=_signal_features(signal),
            file_key=file_key,
            pc_origin="missing" if pc_source == "missing" else pc_origin_of(signal),
            convention=convention, convention_known=convention_known,
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
                pc_source=_inherited_source(parent.pc_source),
                sample_tilt=parent.sample_tilt,
                tilt=parent.tilt,
                azimuthal=parent.azimuthal,
                parent_name=parent_name,
                pc_refinement=_copy_refinement(parent.pc_refinement),
                file_key=parent.file_key,       # a copy is a view of the parent's file
                is_view_of_parent=True,
                pc_origin=parent.pc_origin,
                convention=parent.convention, convention_known=parent.convention_known,
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
            pc_single = parent.pc_single.copy()
            if parent.pc_map is not None:
                pc_map = np.array(window.apply(parent.pc_map), copy=True)
                # the single PC of a map is its mean: the mean of the CUT map, not the
                # parent's (which describes the whole scan)
                pc_single = pc_map.reshape(-1, 3).mean(axis=0)

            entry = CalibrationEntry(
                dataset_name=name,
                detector_shape=parent.detector_shape,
                pc_single=pc_single,
                pc_map=pc_map,
                pc_source=_inherited_source(parent.pc_source),
                sample_tilt=parent.sample_tilt,
                tilt=parent.tilt,
                azimuthal=parent.azimuthal,
                parent_name=parent_name,
                pc_refinement=_copy_refinement(parent.pc_refinement),
                file_key=parent.file_key,       # a copy is a view of the parent's file
                is_view_of_parent=True,
                pc_origin=parent.pc_origin,
                convention=parent.convention, convention_known=parent.convention_known,
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
        self, name: str, pc, source: str = "refined",
        refinement: Optional[dict] = None, on_refined_map: str = "keep",
    ) -> bool:
        """Update the single-PC value for a dataset.

        Parameters
        ----------
        pc : array-like, shape (3,)
        source : str
            Origin of this PC value ("refined", "manual", etc.).
        refinement : dict, optional
            ``{"n_patterns", "method"}`` when this PC was produced by PC
            refinement. Anything else (the default) clears the record: the PC
            stored now no longer comes from a refinement.
        on_refined_map : str
            What a dataset's per-pixel map from a grid calibration (``refined_map``)
            does when a single PC is written: ``"keep"`` (default: the map stays,
            only ``pc_single`` changes - what a propagation to a parent copy wants),
            ``"shift"`` (the map moves rigidly so its mean is the new PC, as a
            header map does) or ``"replace"`` (the map is dropped, the PC is the
            one value). The page that edits the active dataset chooses; see
            :meth:`update_pc_detail`.

        Returns True if the entry existed and was updated.
        """
        return self.update_pc_detail(name, pc, source, refinement, on_refined_map) is not None

    def update_pc_detail(
        self, name: str, pc, source: str = "refined",
        refinement: Optional[dict] = None, on_refined_map: str = "keep",
    ) -> Optional[str]:
        """:meth:`update_pc` that says what happened to the per-pixel map.

        Returns ``None`` when ``name`` is not in the store, else one of
        ``"single"`` (no map: the PC was set), ``"shifted"`` (the map moved so its
        mean is the PC), ``"replaced"`` (the map was dropped) or ``"kept"`` (the
        map stayed; only ``pc_single`` changed, so what the map serves and the
        single PC now disagree).
        """
        if on_refined_map not in ("keep", "shift", "replace"):
            raise ValueError(f"on_refined_map must be keep, shift or replace, got {on_refined_map!r}")
        pc_arr = np.asarray(pc, dtype=float).flatten()[:3]
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                logger.warning("update_pc(%r): not in store", name)
                return None
            prev_source = entry.pc_source
            had_map = entry.pc_map is not None

            if had_map and prev_source == "refined_map" and on_refined_map == "keep":
                # A genuine per-pixel refinement (update_pc_map: Aztec seed / grid
                # calibration) OUTRANKS a global single-PC update. Keep the map AND
                # its "refined_map" provenance so get_detector still serves the
                # per-pixel field and a later refine doesn't mistake it for a
                # shiftable header map. Record the requested single PC for get_pc
                # consumers only.
                entry.pc_single = pc_arr
                action = "kept"
            elif had_map and prev_source == "refined_map" and on_refined_map == "replace":
                # The PC page asked for ONE PC for this dataset (a global refine):
                # the grid map is dropped, so what get_detector serves is that PC.
                entry.pc_map = None
                entry.pc_single = pc_arr
                entry.pc_source = source
                entry.pc_refinement = _copy_refinement(refinement)
                action = "replaced"
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
                if prev_source != "refined_map":
                    # a grid-fitted map stays a grid-fitted map after the shift
                    entry.pc_source = source
                entry.pc_refinement = _copy_refinement(refinement)
                action = "shifted"
            else:
                # No map: plain single-PC update (original behaviour).
                entry.pc_single = pc_arr
                entry.pc_source = source
                entry.pc_refinement = _copy_refinement(refinement)
                action = "single"
        logger.info("update_pc(%r): [%.4f, %.4f, %.4f] source=%s (%s)",
                    name, *pc_arr, source, action)
        return action

    def family_root(self, name: str) -> str:
        """The root ancestor of ``name`` (``parent_name`` followed to its end): the
        raw dataset a chain of copies was made from; ``name`` itself without a parent."""
        with self._lock:
            seen = {name}
            while self._entries.get(name) is not None and self._entries[name].parent_name:
                parent = self._entries[name].parent_name
                if parent in seen:
                    break
                seen.add(parent)
                name = parent
            return name

    def is_ancestor(self, ancestor: str, name: str) -> bool:
        """Is ``ancestor`` on the ``parent_name`` chain above ``name`` (``name`` itself
        excluded)? Stops at a name it has seen, so it ends on a cycle too."""
        with self._lock:
            seen = {name}
            entry = self._entries.get(name)
            while entry is not None and entry.parent_name:
                parent = entry.parent_name
                if parent == ancestor:
                    return True
                if parent in seen:
                    return False
                seen.add(parent)
                entry = self._entries.get(parent)
            return False

    def detach_from_parent(self, name: str) -> None:
        """Make ``name`` a root of its family: its ``parent_name`` link is dropped.

        Its copies stay its copies. Used by ``copy-pc`` before a descendant of ``name``
        takes ``name`` as its source (see there): a family is a tree, never a cycle.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is not None:
                entry.parent_name = None

    def following_family(self, name: str, exclude=()) -> list:
        """The datasets of ``name``'s family that never had a PC of their own.

        The family is every entry with the same root ancestor (``parent_name``
        followed to its end). A member follows when its PC still comes from a
        relative - ``pc_source`` "inherited", "propagated" by an earlier write, or
        "missing" (a copy of a file that had no PC has none of its own either) -
        and so does every copy between it and the root: a copy of a copy whose
        parent was given a PC of its own follows THAT, not the family. The copies
        of ``name`` itself are measured from ``name`` (which has its PC now), not
        from the root. A member
        whose PC was typed, refined, imported or read from a file is its own, and
        so are the names in ``exclude``.
        """
        following = ("inherited", "propagated", "missing")
        with self._lock:
            def chain(n):
                """n, its parent, ... up to and including the root."""
                out, seen = [n], {n}
                while self._entries.get(out[-1]) is not None and self._entries[out[-1]].parent_name:
                    parent = self._entries[out[-1]].parent_name
                    if parent in seen:
                        break
                    out.append(parent)
                    seen.add(parent)
                return out
            root = chain(name)[-1]
            names = []
            for n, e in self._entries.items():
                if n == name or n in exclude or n == root:
                    continue
                path = chain(n)
                if path[-1] != root:
                    continue
                # A copy of ``name`` (however deep) follows what lies between it and
                # ``name``: ``name`` itself has just been given its PC, so it does not
                # count as an independent link. Any other member follows when every
                # copy between it and the root is not independent; the root is the
                # family's own.
                upto = path.index(name) if name in path else len(path) - 1
                if all(self._entries[m].pc_source in following for m in path[:upto] if m in self._entries):
                    names.append(n)
            return names

    def update_pc_map(self, name: str, pc_map: np.ndarray,
                      refinement: Optional[dict] = None,
                      source: str = "refined_map") -> bool:
        """Set per-pixel PC map for a dataset.

        Also updates pc_single to the mean of the map. ``refinement`` is as for
        :meth:`update_pc`: given only when the map was fitted from refined PCs.
        ``source`` is the map's provenance: ``"refined_map"`` (default) for a map
        fitted by a grid calibration, ``"header"`` for the file's own map (the
        Aztec seed), which a later refine moves instead of sparing.
        """
        pc_map = np.asarray(pc_map, dtype=float)
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                logger.warning("update_pc_map(%r): not in store", name)
                return False
            entry.pc_map = pc_map.copy()
            entry.pc_single = pc_map.reshape(-1, 3).mean(axis=0)
            entry.pc_source = source
            entry.pc_refinement = _copy_refinement(refinement)
        logger.info(
            "update_pc_map(%r): shape %s, mean [%.4f, %.4f, %.4f]",
            name, pc_map.shape, *entry.pc_single,
        )
        return True

    def update_geometry(
        self, name: str, sample_tilt: Optional[float] = None,
        tilt: Optional[float] = None, azimuthal: Optional[float] = None,
    ) -> bool:
        """Set the tilts of a dataset's detector (degrees); ``None`` leaves one as it is.

        ``get_detector`` builds the detector every indexer reads from these, so a
        tilt changed on the PC page has to be written here or it never reaches
        indexing. Returns True if the entry existed.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                logger.warning("update_geometry(%r): not in store", name)
                return False
            if sample_tilt is not None:
                entry.sample_tilt = float(sample_tilt)
            if tilt is not None:
                entry.tilt = float(tilt)
            if azimuthal is not None:
                entry.azimuthal = float(azimuthal)
        logger.info("update_geometry(%r): sample_tilt=%s tilt=%s azimuthal=%s",
                    name, sample_tilt, tilt, azimuthal)
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

    def name_of_file(self, file_key: str) -> Optional[str]:
        """The name of the entry registered for the file ``file_key`` (canonical path), or
        ``None``. Views of a dataset (a deepcopy or a crop: ``is_view_of_parent``, they
        carry their parent's ``file_key``) are not the file's own entry. An entry that
        merely took its PC from another file (``copy-pc`` sets ``parent_name``) is."""
        if not file_key:
            return None
        with self._lock:
            for name, entry in self._entries.items():
                if not entry.is_view_of_parent and entry.file_key == file_key:
                    return name
        return None

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

    #: PC sources that are the user's own work (a refine, a grid calibration, a typed
    #: value) and cannot be read from the file again.
    #: ("propagated" is a PC the user typed or refined on a copy of the dataset and
    #: that was passed on to it.)
    USER_MADE_SOURCES = ("refined", "refined_map", "manual", "propagated", "calibration")

    def user_calibrated(self, names) -> list:
        """The ``names`` whose PC is the user's own work (see ``USER_MADE_SOURCES``)."""
        with self._lock:
            return [n for n in names if n in self._entries
                    and self._entries[n].pc_source in self.USER_MADE_SOURCES]

    @classmethod
    def user_made_of(cls, entries: Dict[str, CalibrationEntry]) -> Dict[str, CalibrationEntry]:
        """The entries of a :meth:`snapshot` that hold the user's own PC of a FILE's dataset.

        Views of a dataset (a crop, a copy: ``is_view_of_parent``) are left out: they go with the
        dataset they view, and nothing re-registers them on a later load.
        """
        return {n: e for n, e in entries.items()
                if e.pc_source in cls.USER_MADE_SOURCES and not e.is_view_of_parent}

    @classmethod
    def without_header_maps(cls, entries: Dict[str, CalibrationEntry]) -> Dict[str, CalibrationEntry]:
        """``entries`` (a :meth:`snapshot`, which owns its copies) without the per-pixel maps the file brought along:
        a stash that holds no pattern would otherwise keep a map the file gives again when it is opened. The entry
        says so (``map_dropped``). A map the user made (``refined_map``) cannot be read from the file again and stays."""
        for entry in entries.values():
            if entry.pc_map is not None and entry.pc_source != "refined_map":
                entry.pc_map = None
                entry.map_dropped = True
        return entries

    def restore(self, entries: Dict[str, CalibrationEntry], skip: Optional[str] = None) -> Optional[str]:
        """Re-insert entries produced by :meth:`snapshot` (per-file restore).

        ``skip`` (the freshly re-registered raw dataset) keeps what the file gave it
        - its header PC, its geometry - so a fresh PC is not overwritten by the
        stashed copy. The user's own calibration of it (a refine, a grid calibration,
        a typed PC, see ``USER_MADE_SOURCES``) is carried over, because it cannot be
        read from the file again, unless the FILE now says something else: a file
        that carries a PC today but carried none (or another) when the calibration
        was made - an .osc put beside it - is the newer evidence, and its PC wins.
        Entries are deep-copied on insert so the live store never aliases the
        (now-consumed) stash.

        Returns ``(notice, discarded_pc)``: ``notice`` is what happened to the ``skip``
        entry's PC - ``"user_pc_kept"`` (the user's calibration is back),
        ``"file_pc_wins"`` (the file's own PC replaced a stashed calibration; the
        calibration's PC is ``discarded_pc``, so the user can type it again) or
        ``None`` (nothing to decide, ``discarded_pc`` None).
        """
        import copy
        notice = None
        discarded = None
        shift = None
        # The file's name in the stash may differ from the fresh one in case only (the same file, opened under
        # another spelling: a file system that ignores case): it is the ``skip`` entry, and takes the fresh name.
        same = {n for n in entries if skip is not None and n.casefold() == skip.casefold()}
        with self._lock:
            for name, entry in entries.items():
                if name in same:
                    fresh = self._entries.get(skip)
                    if (fresh is None or entry.pc_source not in self.USER_MADE_SOURCES
                            or tuple(int(v) for v in fresh.detector_shape)
                            != tuple(int(v) for v in entry.detector_shape)):
                        continue
                    if fresh.pc_source != "missing" and not self._same_file_pc(entry.file_pc, fresh.file_pc):
                        notice = "file_pc_wins"
                        discarded = [float(v) for v in np.asarray(entry.pc_single, dtype=float).ravel()[:3]]
                        continue
                    kept = copy.deepcopy(entry)
                    if kept.map_dropped:
                        # the stash holds the PC only; the file's own map (the fresh entry's) moves with it below
                        shift = kept
                    else:
                        fresh.pc_single = kept.pc_single
                        fresh.pc_map = kept.pc_map
                        fresh.pc_source = kept.pc_source
                        fresh.pc_refinement = kept.pc_refinement
                    fresh.sample_tilt = kept.sample_tilt
                    fresh.tilt = kept.tilt
                    fresh.azimuthal = kept.azimuthal
                    notice = "user_pc_kept"
                    continue
                inserted = copy.deepcopy(entry)
                if inserted.parent_name in same:         # a copy of the file named it as the stash spelled it
                    inserted.parent_name = skip
                self._entries[name] = inserted
        if shift is not None:
            # as for an open file: a header map moves rigidly so that its mean is the PC, and a file without a map
            # gets the single PC (``update_pc_detail`` takes the lock, so it runs after the block)
            self.update_pc_detail(skip, shift.pc_single, shift.pc_source, shift.pc_refinement, "shift")
        return notice, discarded

    def restore_pc(self, entries: Dict[str, CalibrationEntry]) -> list:
        """Put the PC of the entries of a :meth:`snapshot` back on the live entries.

        Only what a PC write changes is restored: ``pc_single``, ``pc_map``, ``pc_source`` and
        ``pc_refinement``. Tilts, features, ``file_pc`` and the rest of a live entry stay as they are.
        An entry that is gone, or that now belongs to another file (the store is keyed by the dataset
        name, and two files can share one), is skipped. Returns the names restored.
        """
        import copy
        restored = []
        with self._lock:
            for name, old in entries.items():
                live = self._entries.get(name)
                if live is None or live.file_key != old.file_key:
                    continue
                live.pc_single = copy.deepcopy(old.pc_single)
                live.pc_map = copy.deepcopy(old.pc_map)
                live.pc_source = old.pc_source
                live.pc_refinement = copy.deepcopy(old.pc_refinement)
                restored.append(name)
        for name in restored:
            logger.info("restore_pc(%r): put back to source=%s", name, entries[name].pc_source)
        return restored

    @staticmethod
    def _same_file_pc(before, now) -> bool:
        """True when the file gave the same PC when the calibration was made as now."""
        if before is None or now is None:
            return False
        return bool(np.allclose(np.asarray(before, dtype=float).ravel()[:3],
                                np.asarray(now, dtype=float).ravel()[:3], rtol=0, atol=1e-6))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._entries


# Module-level singleton — import this everywhere
calibration_store = CalibrationStore()
