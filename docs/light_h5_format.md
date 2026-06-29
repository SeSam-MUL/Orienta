# Light `.h5` Format — `result_<stem>_light.h5`

> Compact, MATLAB-compatible export of an EBSD indexing run. Created by
> **Phase Maps → Save as… → Light .h5** (route `POST /api/indexing/export`
> with `format='h5_light'`). Typical size 5–50 MB regardless of how big the
> source dataset was. Designed for sharing with colleagues / MTEX / external
> tools without dragging the multi-GB raw patterns along.

## TL;DR — Why "light"?

The **rich** `.h5` export copies the entire source `.h5oina` (often >25 GB)
and appends an `/Indexing` group. That's great for archival but bad for
sharing:

* multi-GB file size
* inherits HDF5 superblock v2 from the source → MATLAB R2019b and older
  refuse to open it with `H5Fget_obj_count: not a file id`

The **light** export skips the source copy. It writes a fresh file with
HDF5 superblock v0 (the universally readable version) and only the
indexing payload plus the pointers needed to re-link the original. You get
about the same scientific value for ~0.1 % of the disk footprint.

| Property | Light | Rich |
|---|---|---|
| Typical size on a 2k×2k scan with 3 phases | ~10–50 MB | source size + 10 MB |
| Source patterns included? | No (`/SourceReference` points back to source) | Yes (verbatim copy of `/1/EBSD`) |
| HDF5 superblock version | **0** | **2** (inherited from source) |
| MATLAB R2019b compatible? | Yes | Often not |
| Re-importable in this GUI? | Yes (Add file… → Phase Maps) | Yes |

## File-level properties

| HDF5 property | Value |
|---|---|
| Magic header | `89 48 44 46 0D 0A 1A 0A` (HDF5 standard) |
| Superblock version | **0** — forced via `h5py.File(..., libver='earliest')` |
| Offset / length size | 8 bytes (64-bit, supports files >2 GB) |
| Compression | gzip on per-pixel datasets, none on attrs |

## Complete group/dataset tree

```
/
├─ Indexing/                                 (group; root attrs below)
│  ├─ Assignment/                            (multi-phase layout)
│  │  ├─ phase_id            uint8  (R, C)     1-based, 0 = unindexed
│  │  ├─ euler_angles        float32 (R, C, 3) Bunge ZXZ, radians
│  │  ├─ confidence_index    float32 (R, C)    CI of the winning phase
│  │  ├─ band_contrast       uint8  (R, C)     copied from h5oina (if source reachable)
│  │  ├─ pc_x                float32 (R, C)    copied from h5oina (if source reachable)
│  │  ├─ pc_y                float32 (R, C)    copied from h5oina (if source reachable)
│  │  ├─ dd                  float32 (R, C)    detector distance — copied from h5oina
│  │  └─ bands               uint8  (R, C)     #bands detected — copied from h5oina
│  │
│  │  -- OR --   (single-phase runs use the flat layout, no Assignment group)
│  │
│  ├─ phase_id               uint8  (R, C)     single-phase flat layout
│  ├─ euler_angles           float32 (R, C, 3) single-phase flat layout
│  ├─ confidence_index       float32 (R, C)    single-phase flat layout
│  │
│  ├─ PerPhase/                              (multi-phase only)
│  │  ├─ <phase_name>/
│  │  │   ├─ euler_angles    float32 (R, C, 3) "Euler if THIS phase were chosen"
│  │  │   └─ confidence_index float32 (R, C)   "CI assuming THIS phase"
│  │  └─ <other_phase>/ …
│  │
│  ├─ Phases/                                (phase table — name + symmetry)
│  │  ├─ 1/
│  │  │   ├─ @name          str    "Al"
│  │  │   ├─ @space_group   int    225
│  │  │   └─ @point_group   str    "m-3m"
│  │  ├─ 2/                  …
│  │  └─ N/                  …
│  │
│  ├─ X                      float32 (R, C)     sample X in µm = column·step_size_um (≥1.2)
│  ├─ Y                      float32 (R, C)     sample Y in µm = row·step_size_um    (≥1.2)
│  └─ selection_mask         uint8  (R, C)     1 where pixel was indexed, 0 = skipped
│
├─ SourceReference/                          (POINTER, not data)
│  ├─ @description           str
│  ├─ @source_file_path      str    absolute path to the original h5oina
│  ├─ @source_file_name      str    just the basename
│  ├─ @source_file_stem      str    basename without .h5oina
│  └─ @source_size_mb        float  size of the source file when this light was written
│
├─ Detector/                                 (geometry)
│  ├─ pc                     float64 (3,)     pattern center [PCx, PCy, PCz], EMsoft convention
│  ├─ @sample_tilt           float64          degrees, usually 70.0
│  └─ @shape                 int    [H, W]    detector pixel dimensions
│
└─ Documentation/
   ├─ @format_version        str    "1.1"
   ├─ @description           str    human-readable
   └─ README                 str    full README text
```

### Root `/Indexing` attributes

| Attr | Type | Meaning |
|---|---|---|
| `method` | str | `"spherical"`, `"dictionary"`, `"hough"`, … |
| `software` | str | always `"Kikuchipy GUI"` |
| `created` | str | ISO-8601 UTC timestamp |
| `grid_shape` | int[2] | `[R, C]` — rows, cols |
| `format_version` | str | currently `"1.1"` |
| `step_size_um` | float | µm/pixel, present if format_version ≥ 1.1 |
| `source_vendor` | str | `oxford`/`edax`/`bruker`/`unknown` — vendor frame `euler_angles` is in (≥ 1.3) |
| `orientation_reference_frame` | str | human label: `vendor_stored …` or `native …` (≥ 1.3) |

Per-pixel `/Indexing/X` and `/Indexing/Y` (µm) datasets are present from
format_version ≥ 1.2 in **both** layouts. They are the most foolproof way to
place the map on a grid in MTEX (no need to know the step or reshape by hand).
Before 1.2 the interactive *Save as… → Light .h5* path wrote neither the
coordinates nor `step_size_um`, so MTEX couldn't reconstruct the grid.

### `phase_id` convention (important for downstream)

On disk: **1-based**, with **0 = unindexed**. This matches MTEX / Oxford
.ctf and is the inverse of orix's native 0-based convention. When the GUI
reads the file back it shifts: `orix_phase_id = 0 if disk == 0 else disk - 1`.
Treat this as the canonical convention; do not write 0-based phase_ids to a
light file.

### What is **NOT** in a light file

* Raw EBSD patterns (the 27-GB part of a rich .h5)
* EDS counts / spectrum
* Electron images (BSE / FSE / SE)
* Aztec layered images
* Pattern Center per-pixel time series (only the average is in `/Detector/pc`)

If you need any of these, either keep the source `.h5oina` next to the
light file (it's pointed to via `/SourceReference/@source_file_path`) or
generate a Rich .h5 export instead.

## Loading a light file back

Three options:

1. **In this GUI:** Phase Maps → **Add file…** → pick the `.h5`. It lands
   in the Results Gallery with a real `result_id`, so Save as… and
   → Analysis work straight away. Raw patterns / EDS won't be available
   because they aren't in the light file — load the source separately if
   you need them.

2. **In MATLAB / MTEX:** read `/Indexing/Assignment/phase_id`,
   `/Indexing/Assignment/euler_angles`, etc. directly via `h5read`.
   Coordinates + step come straight from the file (format ≥ 1.2):
   ```matlab
   f     = 'foo_light.h5';
   pid   = h5read(f, '/Indexing/Assignment/phase_id');     % single-phase: '/Indexing/phase_id'
   euler = h5read(f, '/Indexing/Assignment/euler_angles'); % Bunge ZXZ, radians
   ci    = h5read(f, '/Indexing/Assignment/confidence_index');
   X     = h5read(f, '/Indexing/X');                       % µm
   Y     = h5read(f, '/Indexing/Y');                       % µm
   step  = h5readatt(f, '/Indexing', 'step_size_um');      % µm/pixel
   % Build the EBSD object from (X(:), Y(:), euler, pid) — no source h5oina needed.
   ```
   Orientation frame (format ≥ 1.3): `euler_angles` is written in the **source
   vendor's stored frame** — the same convention MTEX's default
   `loadEBSD_h5oina` uses — so a raw read lines up with the Aztec solution with
   **no manual rotation**. For Oxford/Bruker data that means our native
   EMsoft/kikuchipy orientations have been right-multiplied by Rz(+90°) about ND
   (the documented Oxford↔EMsoft in-plane difference; verified on SampleB,
   36°→5° vs Aztec). `/Indexing.attrs['source_vendor']` records which vendor
   frame was applied; re-importing the file into this GUI inverts it
   automatically so internal maps stay in the native frame. You may still need
   the usual specimen-surface tilt step in MTEX — that is unrelated to this
   90° fix. The separate `.ang`/`.ctf` exporters are NOT yet frame-corrected
   (tracked follow-up).

3. **In Python / orix:** the GUI's loader is in
   `backend/api/routes/analysis.py::_load_kikuchipy_rich_h5`. It handles
   both the multi-phase (`/Indexing/Assignment/`) and single-phase
   (flat under `/Indexing/`) layouts and returns an `orix.CrystalMap`.

## Version history

| Version | Date | Change |
|---|---|---|
| 1.0 | first light export | original layout |
| 1.1 | 2026-05-04 | added `step_size_um` attr, `band_contrast` / `pc_x` / `pc_y` / `dd` / `bands` quality fields, fail-loud on missing step |
| 1.2 | 2026-06-05 | added per-pixel `/Indexing/X` + `/Indexing/Y` µm datasets; interactive Save as… → Light .h5 now writes `step_size_um` (was batch-helper only) and fails loud when step can't be resolved |
| 1.3 | 2026-06-05 | `euler_angles` written in the source vendor's stored frame (Aztec/MTEX default) instead of native EMsoft/kikuchipy; `/Indexing.attrs` gains `source_vendor` + `orientation_reference_frame`; reader inverts on re-import. Fixes the 90°-about-ND offset vs Aztec (Oxford `* Rz(+90°)`). |

## Where this format is defined in code

* Writer (Light branch in the export route):
  [`backend/api/routes/indexing.py`](../backend/api/routes/indexing.py) — search for `elif fmt == "h5_light":`
* Writer (Light helper for batch workflow):
  [`backend/api/services/result_exporter.py::export_result_h5_light`](../backend/api/services/result_exporter.py)
* Reader:
  [`backend/api/routes/analysis.py::_load_kikuchipy_rich_h5`](../backend/api/routes/analysis.py)
* Re-import as gallery entry:
  [`backend/api/routes/indexing.py`](../backend/api/routes/indexing.py) — search for `@router.post("/import-h5")`
