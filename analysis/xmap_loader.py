"""One crystal-map loader for the Analysis page and the batch.

:func:`load_xmap_file` is the only place that turns a file into an orix ``CrystalMap``; ``POST
/api/analysis/load`` and :class:`analysis.batch_processor.BatchProcessor` both call it, so a file that opens
on the page opens in a batch, and the other way round.

Formats
-------
* **Aztec ``.h5oina``** (Oxford, indexed in Aztec): read from ``/1/EBSD`` by :func:`load_aztec_h5oina`
  (orix cannot read it).
* **Orienta's own result ``.h5``** (rich and light exports): :func:`load_orienta_result_h5`.
* **``.ang`` / ``.ctf``** and everything else orix knows: ``orix.io.load``; if that fails the Orienta
  reader gets a try, and both reasons are in the error (the message the Analysis route always gave).

Aztec ``.h5oina`` in detail
---------------------------
* **Slice.** Only slice ``"1"`` is read, as everywhere else in Orienta (``meta["n_slices"]`` says how many
  the file has). A file without slice ``"1"`` is refused.
* **Orientations.** ``/1/EBSD/Data/Euler`` (radians, Bunge) exactly as stored. That is Orienta's canonical
  frame (CS1, raw Aztec Euler, what MTEX's default ``h5oina`` import gives):
  ``backend.api.services.orientation_frame.oxford_frame_adapter`` is the identity, so nothing is applied.
  ``analysis/`` is imported by the backend, never the other way round, so the Aztec reader does not call the
  frame service; ``tests/test_xmap_loader.py`` pins the identity so a change there cannot go unnoticed.
  The one exception in this module is :func:`load_orienta_result_h5`, moved here from the route: for result
  files that name the vendor frame they were exported in, it imports
  ``backend.api.services.orientation_frame`` *inside the function* (never at import time; a test pins that
  importing this module does not import ``backend``).
* **Phases.** ``/1/EBSD/Data/Phase``: 0 = not indexed (orix id -1), n >= 1 is orix id n - 1 and the header
  group ``/1/EBSD/Header/Phases/<n>``. Name, space group and cell come from that group. The point group is
  the one of the space group number (``crystal_symmetry.FramePhase``, which also gives monoclinic phases
  their two-fold axis). A phase that pixels use but whose space group is missing or outside 1..230 is an
  error, never a guess; a header phase without any pixel and without a usable symmetry is left out
  (``meta["skipped_phases"]``).
* **Grid.** ``X``/``Y`` from the file; the step is ``X Step`` of the header (``meta["step_from_file"]``),
  rounded to 7 significant digits (the header stores float32: 0.15 stays 0.15). A missing step, unequal
  ``X Step`` and ``Y Step``, a pixel count that is not ``X Cells * Y Cells``, and a layout other than
  "rows of ``X Cells`` pixels, X fastest, every row at the same X" (column-major files, hex grids with
  shifted rows) are refused, each with the file name.
* **Band contrast.** ``/1/EBSD/Data/Band Contrast`` (native, 0..255) becomes ``xmap.prop["bc"]``; without
  that dataset the key is absent and the band-contrast analyses say so. One value per pixel is required.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AZTEC_SLICE = "1"


class _NotAztecLayout(ValueError):
    """The file is not an Aztec ``/1/EBSD/Data`` + ``Header`` file (so orix gets its turn)."""


def step_of_xmap(xmap) -> float:
    """The step of a CrystalMap in um: ``dx``; from the coordinates if it has none; 1.0 if nothing says."""
    dx = getattr(xmap, "dx", None)
    if dx is not None:
        return float(dx)
    x = getattr(xmap, "x", None)
    if x is not None and len(x) > 1:
        return float(np.median(np.diff(np.unique(x))))
    return 1.0


def load_xmap_file(path: str) -> Tuple[Any, float, Dict[str, Any]]:
    """Read a crystal-map file.

    Args:
        path: ``.h5oina`` (Aztec), Orienta result ``.h5``, ``.ang``, ``.ctf`` or any other orix format

    Returns:
        ``(xmap, step_um, meta)``. ``meta["format"]`` is ``"aztec_h5oina"``, ``"orienta_result_h5"`` or
        ``"orix"``; ``meta["step_from_file"]`` is True when the step is a value the file states (the Aztec,
        .ang or .ctf header), False when it was taken from the coordinates of another file (a caller may then
        ask another source when the step is the unit-pixel 1.0). A hexagonal ``.ang`` adds ``hex_grid`` (see
        :func:`analysis.ang_hexgrid.load_ang_xmap`). An Aztec file adds ``phases``, ``skipped_phases``, ``slice``,
        ``n_slices`` and ``has_native_bc``.

    Raises:
        AngGridRefused: a HexGrid ``.ang`` that is not a regular hexagonal scan (``code`` names why)
        ValueError: the file cannot be read; the message says why (for orix formats: what orix and the
            Orienta reader each said)
    """
    if str(path).lower().endswith(".ang"):
        # The header decides the grid first (analysis/ang_hexgrid.py): a HexGrid file is kept as a
        # hexagonal scan (meta["hex_grid"], the plain grid dict) or refused by name (AngGridRefused, a
        # ValueError with a stable ``code``); it is never read as a square map. A SqrGrid file goes to orix.
        from analysis.ang_hexgrid import AngGridRefused, load_ang_xmap
        try:
            xmap, grid = load_ang_xmap(path)
        except AngGridRefused:
            raise
        except Exception as orix_err:
            raise ValueError(f"orix.io.load failed: {orix_err}") from orix_err
        if grid is not None:
            return xmap, float(grid["step_x"]), {"format": "orix", "step_from_file": True, "hex_grid": grid}
        return xmap, step_of_xmap(xmap), {"format": "orix", "step_from_file": True}

    aztec_err = None
    if str(path).lower().endswith(".h5oina"):
        try:
            return load_aztec_h5oina(path)
        except _NotAztecLayout as not_aztec:
            aztec_err = str(not_aztec)

    from orix.io import load as orix_load
    try:
        xmap = orix_load(path)
        fmt = "orix"
    except Exception as orix_err:
        try:
            xmap = load_orienta_result_h5(path)
            fmt = "orienta_result_h5"
        except Exception as our_err:
            prefix = f"Aztec .h5oina reader: {aztec_err}\n" if aztec_err else ""
            raise ValueError(
                f"{prefix}orix.io.load failed: {orix_err}\n"
                f"Custom .h5 fallback failed: {our_err}"
            ) from our_err
    # .ang and .ctf state the step in their header (XSTEP / XStep) and orix builds the coordinates from it, so a
    # step of exactly 1.0 there is a measurement; other files may carry no coordinates (unit-pixel placeholder)
    stated = fmt == "orix" and str(path).lower().endswith((".ang", ".ctf"))
    return xmap, step_of_xmap(xmap), {"format": fmt, "step_from_file": stated}


def _header_step(hd, name: str, path: str) -> float:
    if name not in hd:
        raise ValueError(f"{path}: the header has no '{name}'; the step size of the map is unknown")
    # the header stores float32: keep the digits that were meant (0.15, not 0.15000000596046448)
    return float(f"{float(_scalar(hd[name])):.7g}")


def _scalar(ds):
    v = np.asarray(ds[()])
    return v.reshape(-1)[0] if v.size else None


def _text(ds) -> str:
    v = _scalar(ds)
    return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)


def load_aztec_h5oina(path: str):
    """Build a CrystalMap from an Aztec-indexed ``.h5oina`` (see the module docstring for what is read).

    Returns:
        ``(xmap, step_um, meta)`` as :func:`load_xmap_file`
    """
    import h5py
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from crystal_symmetry import FramePhase

    try:
        f = h5py.File(path, "r")
    except OSError as e:
        raise _NotAztecLayout(f"not an HDF5 file ({e})") from e
    with f:
        slices = sorted(k for k in f.keys() if k.isdigit())
        if f"{AZTEC_SLICE}/EBSD/Data" not in f or f"{AZTEC_SLICE}/EBSD/Header" not in f:
            if not slices or "Indexing" in f:
                raise _NotAztecLayout("no /1/EBSD/Data and /1/EBSD/Header")
            raise ValueError(
                f"{path}: the file has no slice '{AZTEC_SLICE}' with EBSD data (slices: {slices}); "
                "only slice 1 is read")
        d = f[f"{AZTEC_SLICE}/EBSD/Data"]
        hd = f[f"{AZTEC_SLICE}/EBSD/Header"]
        for need in ("Euler", "Phase", "X", "Y"):
            if need not in d:
                raise ValueError(f"{path}: /1/EBSD/Data/{need} is missing; this is not an indexed Aztec map")
        euler = np.asarray(d["Euler"], dtype=np.float64).reshape(-1, 3)
        phase = np.asarray(d["Phase"]).astype(np.int64).reshape(-1)
        x = np.asarray(d["X"], dtype=np.float64).reshape(-1)
        y = np.asarray(d["Y"], dtype=np.float64).reshape(-1)
        bc = np.asarray(d["Band Contrast"]).reshape(-1) if "Band Contrast" in d else None
        if not (euler.shape[0] == phase.size == x.size == y.size):
            raise ValueError(f"{path}: Euler, Phase, X and Y do not have the same number of pixels")
        if bc is not None and bc.size != phase.size:
            raise ValueError(
                f"{path}: Band Contrast has {bc.size} values for {phase.size} pixels; one value per pixel is needed")

        step_x = _header_step(hd, "X Step", path)
        step_y = _header_step(hd, "Y Step", path)
        if not np.isclose(step_x, step_y, rtol=1e-4):
            raise ValueError(
                f"{path}: X Step ({step_x}) and Y Step ({step_y}) differ; the analysis has one step size")
        if "X Cells" not in hd or "Y Cells" not in hd:
            raise ValueError(f"{path}: the header has no 'X Cells' / 'Y Cells'; the grid of the map is unknown")
        nx, ny = int(_scalar(hd["X Cells"])), int(_scalar(hd["Y Cells"]))
        if nx * ny != phase.size:
            raise ValueError(
                f"{path}: the header says {nx} x {ny} cells but the data has {phase.size} pixels")
        # rows of nx pixels, X running fastest, every row over the same X and at one Y: anything else (a
        # column-major file, a hex grid whose rows are shifted) would be read in the wrong order
        tol = 1e-3 * step_x
        xs, ys = x.reshape(ny, nx), y.reshape(ny, nx)
        if not (np.allclose(xs, xs[0], atol=tol, rtol=0) and np.allclose(ys, ys[:, :1], atol=tol, rtol=0)):
            raise ValueError(
                f"{path}: the pixel positions are not a square grid of {ny} rows of {nx} pixels with X running "
                "fastest (column-major or shifted rows, e.g. a hex grid); only that layout is read")

        used = {int(v) for v in np.unique(phase) if v > 0}
        phases_grp = hd["Phases"] if "Phases" in hd else {}
        header_ids = sorted(int(k) for k in phases_grp.keys() if str(k).isdigit())
        missing = sorted(used - set(header_ids))
        if missing:
            raise ValueError(
                f"{path}: pixels belong to phase {missing[0]}, which the header does not describe "
                f"(header phases: {header_ids})")

        ids, built, skipped = [], [], []
        for k in header_ids:
            g = phases_grp[str(k)]
            name = _text(g["Phase Name"]) if "Phase Name" in g else f"phase_{k}"
            sg = int(_scalar(g["Space Group"])) if "Space Group" in g else 0
            if not 1 <= sg <= 230:
                if k in used:
                    raise ValueError(
                        f"{path}: phase {k} ({name}) has no usable space group (header says {sg}); "
                        "its point group, and with it every misorientation, would be a guess")
                skipped.append(name)
                continue
            structure = None
            if "Lattice Dimensions" in g and "Lattice Angles" in g:
                dims = np.asarray(g["Lattice Dimensions"][()], dtype=float).reshape(-1)
                ang = np.rad2deg(np.asarray(g["Lattice Angles"][()], dtype=float).reshape(-1))
                if dims.size == 3 and ang.size == 3 and (dims > 0).all() and (ang > 0).all():
                    from diffpy.structure import Lattice, Structure
                    structure = Structure(lattice=Lattice(*dims, *ang))
            ids.append(k - 1)
            built.append(FramePhase(name=name, space_group=sg, structure=structure))
        if not built:
            raise ValueError(f"{path}: the header lists no phase with a usable space group")
        phase_list = PhaseList(phases=dict(zip(ids, built)))

    # Aztec: 0 = not indexed, n >= 1 = header phase n; orix: -1 = not indexed, id n - 1
    xmap = CrystalMap(
        rotations=Rotation.from_euler(euler), phase_id=phase - 1, x=x, y=y, phase_list=phase_list,
        prop={"bc": bc.astype(np.float32)} if bc is not None else None, scan_unit="um",
    )
    if tuple(xmap.shape) != (ny, nx):
        raise ValueError(f"{path}: the map came out as {tuple(xmap.shape)}, the header says {(ny, nx)}")
    meta = {
        "format": "aztec_h5oina", "step_from_file": True, "slice": AZTEC_SLICE, "n_slices": len(slices),
        "phases": [p.name for p in built], "skipped_phases": skipped, "has_native_bc": bc is not None,
    }
    return xmap, step_x, meta


def load_orienta_result_h5(path: str):
    """Build a CrystalMap from our own .h5 export schema.

    Handles three export layouts produced by the GUI (see result_exporter.py
    and indexing.py /api/indexing/export):

    **Rich/Light multi-phase**  ``result_<stem>.h5`` / ``..._light.h5``:
       - ``/Indexing/Assignment/phase_id``      (uint8, 0=not indexed, 1..N=phase)
       - ``/Indexing/Assignment/euler_angles``  (float32, radians, Bunge)
       - ``/Indexing/Assignment/confidence_index``
       - ``/Indexing/Phases/<1..N>`` with attrs.name

    **Rich single-phase**       ``result_<stem>.h5``:
       - same fields, but written FLAT under ``/Indexing/`` (no Assignment
         sub-group) because indexing.py's single-phase branch skips the
         consensus group when there's only one phase. The writer/reader
         used to disagree on this layout — see the Assignment-or-flat
         fallback below.

    orix.io.load doesn't recognise any of these; this walks the h5 directly
    and returns a CrystalMap that Analysis / Phase Map can consume.

    Moved here unchanged from ``backend/api/routes/analysis.py`` (``_load_kikuchipy_rich_h5``, which is
    still that module's name for it). The one reach into ``backend`` below is lazy and only for files that
    name the vendor frame they were exported in.
    """
    import h5py
    import numpy as np
    from orix.crystal_map import CrystalMap, PhaseList
    from crystal_symmetry import FramePhase as Phase
    from orix.quaternion import Rotation

    with h5py.File(path, "r") as f:
        idx = f.get("Indexing")
        if idx is None:
            raise ValueError("file has no /Indexing group — not a GUI export")
        asg = idx.get("Assignment")
        if asg is None:
            # Single-phase rich-h5 layout: datasets sit flat under /Indexing/
            # rather than nested in /Indexing/Assignment/. Treat /Indexing/
            # itself as the assignment group — the dataset names are
            # identical, just one level up. Quality fields are absent in
            # this layout but the loop below handles that gracefully.
            if "phase_id" not in idx or "euler_angles" not in idx:
                raise ValueError(
                    "/Indexing exists but neither /Indexing/Assignment nor "
                    "flat phase_id+euler_angles datasets are present"
                )
            asg = idx

        phase_id = np.asarray(asg["phase_id"])  # 0 = not indexed, 1..N = phase
        euler = np.asarray(asg["euler_angles"])  # shape (rows, cols, 3) in radians
        ci = np.asarray(asg.get("confidence_index", np.zeros_like(phase_id, dtype=np.float32)))
        # Step size in µm — exporter (format_version >= 1.1) writes this
        # into /Indexing.attrs. Older files don't carry it; we set 0.0
        # here and let the caller's _fallback_step_size pick up the
        # active EBSD signal's axes manager. Keeping 0.0 (not 1.0) lets
        # the load_xmap detection logic distinguish "file says 1 µm"
        # from "file doesn't say".
        try:
            step_size_um = float(idx.attrs.get("step_size_um", 0.0))
        except (TypeError, ValueError):
            step_size_um = 0.0
        # Orientation export-frame tag (format_version >= 1.3). When present,
        # the stored Euler is in the vendor's frame (Aztec/MTEX) and must be
        # inverted back to our native (EMsoft/kikuchipy) frame so the working
        # map matches freshly-indexed results and the renderer. Absent on
        # older files, which were always written native.
        _src_vendor = idx.attrs.get("source_vendor", None)
        if isinstance(_src_vendor, bytes):
            _src_vendor = _src_vendor.decode("utf-8", errors="replace")
        # Optional per-pixel quality fields carried through from the source
        # h5oina by export_result_h5_light. Older light files don't have
        # them; in that case the prop key is simply absent.
        quality_keys = ("band_contrast", "mad", "pc_x", "pc_y", "dd", "bands")
        quality_data: dict = {
            k: np.asarray(asg[k]) for k in quality_keys if k in asg
        }

        grid_shape = phase_id.shape
        n_pixels = int(np.prod(grid_shape))

        # Per-phase CI maps — loaded for the "CI — <phase>" and uncertainty
        # views in Phase Map. Lives under /Indexing/PerPhase/<name>/ in the
        # h5 schema written by result_exporter.export_result_h5_light /
        # export_result_h5. Absent on .ang/.ctf (single-CI formats).
        per_phase_data: dict = {}
        per_phase_grp = idx.get("PerPhase")
        if per_phase_grp is not None:
            for pname in per_phase_grp:
                sub = per_phase_grp[pname]
                entry: dict = {}
                if "confidence_index" in sub:
                    entry["ci"] = np.array(sub["confidence_index"]).astype(np.float32)
                if "euler_angles" in sub:
                    pe = np.array(sub["euler_angles"]).astype(np.float32)
                    if _src_vendor and str(_src_vendor) not in ("", "unknown"):
                        from backend.api.services.orientation_frame import (
                            from_vendor_export_frame as _from_vendor,
                        )
                        _shp = pe.shape
                        pe = _from_vendor(
                            Rotation.from_euler(pe.reshape(-1, 3).astype(np.float64)),
                            str(_src_vendor),
                        ).to_euler().reshape(_shp).astype(np.float32)
                    entry["euler"] = pe
                if entry:
                    per_phase_data[pname] = entry

        # Phase list — pick up space_group / point_group attrs so the
        # rebuilt Phase objects carry the crystallographic symmetry. With
        # them missing, orix treats every phase as triclinic and IPF
        # coloring degenerates into noise, so the export writes these
        # attrs.
        phases_grp = idx.get("Phases") or idx.get("phases")
        phase_names = []
        phase_sgs: list = []      # int or None
        phase_pgs: list = []      # str or None
        phase_lattices: list = []  # [a, b, c, alpha, beta, gamma] angstrom/deg or None
        if phases_grp is not None:
            # Keys are "1", "2", ... — iterate in numeric order
            for key in sorted(phases_grp.keys(), key=lambda k: int(k) if k.isdigit() else 0):
                nm = phases_grp[key].attrs.get("name", f"phase_{key}")
                if isinstance(nm, bytes):
                    nm = nm.decode("utf-8", errors="replace")
                phase_names.append(str(nm))
                sg = phases_grp[key].attrs.get("space_group", None)
                pg = phases_grp[key].attrs.get("point_group", None)
                if isinstance(pg, bytes):
                    pg = pg.decode("utf-8", errors="replace")
                phase_sgs.append(int(sg) if sg is not None else None)
                phase_pgs.append(str(pg) if pg else None)
                # format >= 1.4: the lattice the exporter recorded (angstrom, deg)
                lat = phases_grp[key].attrs.get("lattice_constants", None)
                phase_lattices.append(
                    [float(v) for v in np.asarray(lat).ravel()]
                    if lat is not None and np.asarray(lat).size == 6 else None)
        if not phase_names:
            # Fall back: derive from max phase_id
            n = int(phase_id.max()) if phase_id.size else 0
            phase_names = [f"phase_{i+1}" for i in range(n)]
            phase_sgs = [None] * n
            phase_pgs = [None] * n

    # Build flat arrays
    phase_id_flat = phase_id.flatten().astype(int)
    euler_flat = euler.reshape(n_pixels, 3)
    ci_flat = ci.flatten().astype(np.float32)

    # Pixel coordinates → µm coordinates so xmap.step_sizes returns the
    # real value. Without this every downstream consumer (EBSDDataset,
    # Phase Map, PC Refinement) reconstructs an axes manager with
    # scale=1.0 even though the file knows the real step. When the file
    # didn't carry a step (older exports), use 1.0 unit-coordinates and
    # let the caller substitute from the active EBSD signal.
    scale = step_size_um if step_size_um > 0 else 1.0
    y_coords, x_coords = np.mgrid[0:grid_shape[0], 0:grid_shape[1]]
    x_flat = (x_coords.flatten() * scale).astype(float)
    y_flat = (y_coords.flatten() * scale).astype(float)

    # Phase list — Phase 0 represents "not indexed"; orix handles that via id -1.
    # Build each Phase with its persisted symmetry (space_group preferred,
    # point_group fallback) so IPF coloring has something to reduce against.
    built_phases = []
    for i, nm in enumerate(phase_names):
        sg = phase_sgs[i] if i < len(phase_sgs) else None
        pg = phase_pgs[i] if i < len(phase_pgs) else None
        ph = None
        if sg is not None:
            try: ph = Phase(name=nm, space_group=int(sg))
            except Exception: ph = None
        if ph is None and pg:
            try: ph = Phase(name=nm, point_group=str(pg))
            except Exception: ph = None
        if ph is None:
            ph = Phase(name=nm)
        built_phases.append(ph)
    phase_list = PhaseList(built_phases)

    # Remap: our 0 = unindexed → orix's phase_id convention usually uses 0..N-1.
    # Shift real phases by -1 so orix phase 0 = first real phase.
    # Unindexed pixels get -1 which orix treats as "not indexed".
    orix_phase_id = np.where(phase_id_flat == 0, -1, phase_id_flat - 1)

    rotations = Rotation.from_euler(euler_flat, degrees=False)
    # Invert the export-frame rotation so re-imported maps return to native.
    if _src_vendor and str(_src_vendor) not in ("", "unknown"):
        from backend.api.services.orientation_frame import from_vendor_export_frame
        rotations = from_vendor_export_frame(rotations, str(_src_vendor))
    # Map disk-key → xmap.prop key. 'band_contrast' on disk shows up as
    # 'bc' in prop because that's the convention the analysis routes use.
    DISK_TO_PROP = {
        "band_contrast": "bc",
        "mad": "mad",
        "pc_x": "pc_x",
        "pc_y": "pc_y",
        "dd": "dd",
        "bands": "bands",
    }
    prop_dict = {"ci": ci_flat}
    for disk_key, prop_key in DISK_TO_PROP.items():
        arr = quality_data.get(disk_key)
        if arr is not None and arr.size == n_pixels:
            prop_dict[prop_key] = arr.flatten().astype(np.float32)
    xmap = CrystalMap(
        rotations=rotations,
        phase_id=orix_phase_id,
        x=x_flat,
        y=y_flat,
        phase_list=phase_list,
        prop=prop_dict,
        scan_unit="um",
    )
    # Stash per-phase CI maps on the xmap so load_xmap can mirror them
    # onto the EBSDDataset. Attribute name matches the one indexing
    # already uses on IndexingResult.metadata['per_phase_data'].
    try:
        xmap._per_phase_data = per_phase_data or None
        # Re-import restores what the file recorded about its own grid and
        # phases, so a re-exported .ang needs nothing from a loaded source.
        xmap._step_size_um = step_size_um if step_size_um > 0 else None
        xmap._recorded_lattices = {
            i: lat for i, lat in enumerate(phase_lattices) if lat is not None}
    except Exception:
        pass
    return xmap
