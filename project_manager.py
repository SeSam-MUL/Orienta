"""Project Manager — save/load Orienta session as .kgproj bundle.

A .kgproj is a directory containing:
  project.json          — metadata (names, methods, step sizes, phase info)
  results/
    phase_map_N.npy   — 2D phase ID array for entry N
    xmap_N.h5         — orix CrystalMap for entry N (optional)
    scores_N.npy      — confidence scores array (optional)
    mask_N.npy        — selection mask (optional)

Usage
-----
    from project_manager import GalleryEntry, save_project, load_project

    entry = GalleryEntry(name="Fe Hough 14:30", method="Hough", phase_data=pd)
    save_project("MyProject.kgproj", [entry])
    meta, gallery = load_project("MyProject.kgproj")
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_VERSION = "1.0"


@dataclass
class GalleryEntry:
    """A single indexing result stored in the Phase Maps gallery.

    Attributes
    ----------
    name : str
        User-assigned display name shown in the gallery list.
    method : str
        Indexing method: "Hough", "Dictionary", "Spherical", or "Comparison".
    phase_data : object
        PhaseMapData from tools.phase_map_generator — always present.
    indexing_result : object, optional
        IndexingResult from indexing_controller — enables Pattern Match Viewer
        and IPF coloring. May be None for consensus/comparison maps.
    mean_score : float, optional
        Mean confidence / NCC score across indexed pixels.
    timestamp : str
        ISO timestamp string ("HH:MM:SS") when result was created.
    """
    name: str
    method: str
    phase_data: object
    indexing_result: object = None
    mean_score: Optional[float] = None
    timestamp: str = ""
    params: Optional[Dict] = None    # indexing config summary (n_bands, t_sigma, …)


def save_project(
    path: str,
    gallery: List[GalleryEntry],
    h5_file_path: str = "",
) -> Path:
    """Save all gallery entries to a .kgproj directory bundle.

    Parameters
    ----------
    path : str
        Destination directory path, e.g. ``"/data/MyProject.kgproj"``.
        Created if it does not exist.
    gallery : List[GalleryEntry]
        All entries to persist.
    h5_file_path : str
        Reference path to the source H5OINA file (stored for reference,
        the file itself is NOT copied into the bundle).

    Returns
    -------
    Path
        The created project directory.
    """
    proj_dir = Path(path)
    proj_dir.mkdir(parents=True, exist_ok=True)
    results_dir = proj_dir / "results"
    results_dir.mkdir(exist_ok=True)

    saved_entries = []
    for i, entry in enumerate(gallery):
        meta: Dict = {
            "name": entry.name,
            "method": entry.method,
            "mean_score": entry.mean_score,
            "timestamp": entry.timestamp,
            # Files (populated below)
            "phase_map_file": None,
            "xmap_file": None,
            "scores_file": None,
            "mask_file": None,
            "original_shape": None,
            "method_enum": None,
            # PhaseMapData fields
            "step_x": 1.0,
            "step_y": 1.0,
            "title": entry.name,
            "phases": [],
        }

        # ── PhaseMapData ────────────────────────────────────────────────
        if entry.phase_data is not None:
            pd = entry.phase_data
            meta["step_x"] = float(pd.step_x)
            meta["step_y"] = float(pd.step_y)
            meta["title"] = pd.title
            meta["phases"] = [
                {"phase_id": p.phase_id, "name": p.name, "color": p.color}
                for p in pd.phases
            ]
            fname = f"phase_map_{i}.npy"
            np.save(results_dir / fname, pd.phase_map_2d)
            meta["phase_map_file"] = fname

        # ── IndexingResult (optional) ───────────────────────────────────
        if entry.indexing_result is not None:
            res = entry.indexing_result
            meta["method_enum"] = res.method.value
            meta["original_shape"] = list(res.original_shape)

            # xmap → orix HDF5
            try:
                from orix.io import save as orix_save
                xmap_file = f"xmap_{i}.h5"
                orix_save(str(results_dir / xmap_file), res.xmap)
                meta["xmap_file"] = xmap_file
            except Exception as exc:
                logger.warning(f"Could not save xmap for entry {i}: {exc}")

            # Confidence scores
            if res.confidence_scores is not None:
                sf = f"scores_{i}.npy"
                np.save(results_dir / sf, res.confidence_scores)
                meta["scores_file"] = sf

            # Selection mask
            if res.selection_mask is not None:
                mf = f"mask_{i}.npy"
                np.save(results_dir / mf, res.selection_mask)
                meta["mask_file"] = mf

        saved_entries.append(meta)

    project_meta = {
        "version": PROJECT_VERSION,
        "created": datetime.now().isoformat(),
        "h5_file_path": h5_file_path,
        "gallery": saved_entries,
    }
    with open(proj_dir / "project.json", "w", encoding="utf-8") as f:
        json.dump(project_meta, f, indent=2)

    logger.info(f"Project saved to {proj_dir} ({len(gallery)} entries)")
    return proj_dir


def load_project(path: str) -> Tuple[Dict, List[GalleryEntry]]:
    """Load a .kgproj directory bundle.

    Parameters
    ----------
    path : str
        Path to the .kgproj directory.

    Returns
    -------
    (meta, gallery)
        meta : dict
            Keys: ``version``, ``created``, ``h5_file_path``.
        gallery : List[GalleryEntry]
            Reconstructed gallery entries (xmap loaded lazily from HDF5).
    """
    proj_dir = Path(path)
    results_dir = proj_dir / "results"

    json_path = proj_dir / "project.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Not a valid .kgproj folder: {proj_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    from tools.phase_map_generator import PhaseMapData, PhaseInfo

    gallery: List[GalleryEntry] = []
    for i, em in enumerate(meta.get("gallery", [])):

        # ── PhaseMapData ────────────────────────────────────────────────
        phase_data = None
        pm_file = em.get("phase_map_file")
        if pm_file and (results_dir / pm_file).exists():
            arr = np.load(results_dir / pm_file)
            phases = [
                PhaseInfo(phase_id=p["phase_id"], name=p["name"], color=p["color"])
                for p in em.get("phases", [])
            ]
            phase_data = PhaseMapData(
                phase_map_2d=arr,
                phases=phases,
                step_x=em.get("step_x", 1.0),
                step_y=em.get("step_y", 1.0),
                title=em.get("title", em.get("name", f"Result {i + 1}")),
            )

        # ── IndexingResult (optional) ───────────────────────────────────
        indexing_result = None
        xmap_file = em.get("xmap_file")
        if xmap_file and (results_dir / xmap_file).exists():
            try:
                from orix.io import load as orix_load
                from indexing_controller import IndexingResult, IndexingMethod

                xmap = orix_load(str(results_dir / xmap_file))

                method_str = em.get("method_enum", "hough")
                try:
                    method = IndexingMethod(method_str)
                except ValueError:
                    method = IndexingMethod.HOUGH

                scores = None
                sf = em.get("scores_file")
                if sf and (results_dir / sf).exists():
                    scores = np.load(results_dir / sf)

                original_shape = tuple(em.get("original_shape") or xmap.shape)

                mask = None
                mf = em.get("mask_file")
                if mf and (results_dir / mf).exists():
                    mask = np.load(results_dir / mf)
                else:
                    mask = np.ones(original_shape, dtype=bool)

                indexing_result = IndexingResult(
                    xmap=xmap,
                    selection_mask=mask,
                    original_shape=original_shape,
                    method=method,
                    confidence_scores=scores,
                )
            except Exception as exc:
                logger.warning(f"Could not load xmap for entry {i}: {exc}")

        gallery.append(GalleryEntry(
            name=em.get("name", f"Result {i + 1}"),
            method=em.get("method", "Unknown"),
            phase_data=phase_data,
            indexing_result=indexing_result,
            mean_score=em.get("mean_score"),
            timestamp=em.get("timestamp", ""),
        ))

    logger.info(f"Project loaded from {proj_dir} ({len(gallery)} entries)")
    return meta, gallery
