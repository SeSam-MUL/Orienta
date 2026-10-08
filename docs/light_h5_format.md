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
│  ├─ Phases/                                (phase table — name, symmetry, lattice)
│  │  ├─ 1/
│  │  │   ├─ @name                    str      "Al"
│  │  │   ├─ @point_group             str      "m-3m"
│  │  │   ├─ @space_group             int      225          (≥ 1.4, if known)
│  │  │   ├─ @space_group_symbol      str      "Fm-3m"      (≥ 1.4, if known)
│  │  │   ├─ @lattice_constants       float[6] a, b, c, alpha, beta, gamma (≥ 1.4)
│  │  │   ├─ @lattice_length_unit     str      "angstrom"   (a, b, c; angles in degrees)
│  │  │   ├─ @lattice_source          str      where the lattice was read from
│  │  │   ├─ @crystal_reference_frame str      frame of the Euler angles (≥ 1.4)
│  │  │   ├─ @sht_file / @sht_path    str      spherical results only
│  │  │   └─ @lattice_status          str      "unknown" if no lattice was found
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
├─ Acquisition/                              (≥ 1.4; only if the source header has the values)
│  ├─ @Scanning Rotation Angle         float    radians, verbatim from the h5oina header
│  ├─ @Specimen Orientation Euler      float[3] radians, verbatim
│  ├─ @Tilt Angle                      float    radians, verbatim (sample tilt)
│  ├─ @header_values_unit              str      "radians"
│  ├─ @applied_to_euler_angles         int      0 — none of the above is applied
│  ├─ @applied_to_euler_angles_note    str      what a reader has to do
│  ├─ @scanning_rotation_angle_deg, @sample_tilt_deg,
│  │  @specimen_orientation_euler_deg, @detector_orientation_euler_deg,
│  │  @detector_tilt_deg, @working_distance_mm, @beam_voltage_kv   (same facts in
│  │                                          degrees / mm / kV; each only if known)
│  └─ @source                str    "h5oina /1/EBSD/Header"
│
├─ Detector/                                 (geometry)
│  ├─ pc                     float64 (3,)     pattern center [PCx, PCy, PCz], Bruker convention (the one kikuchipy uses)
│  ├─ @sample_tilt           float64          degrees, usually 70.0
│  └─ @shape                 int    [H, W]    detector pixel dimensions
│
└─ Documentation/
   ├─ @format_version        str    "1.4"
   ├─ @description           str    human-readable
   └─ README                 str    full README text
```

### Root `/Indexing` attributes

| Attr | Type | Meaning |
|---|---|---|
| `method` | str | `"spherical"`, `"dictionary"`, `"hough"`, … |
| `software` | str | always `"Orienta"` |
| `created` | str | ISO-8601 UTC timestamp |
| `grid_shape` | int[2] | `[R, C]` — rows, cols |
| `format_version` | str | currently `"1.4"` |
| `step_size_um` | float | µm/pixel, present if format_version ≥ 1.1 |
| `source_vendor` | str | `oxford`/`edax`/`bruker`/`unknown` — vendor frame `euler_angles` is in (≥ 1.3) |
| `orientation_reference_frame` | str | human label: `vendor_stored …` or `native …` (≥ 1.3) |
| `scan_row_offset` | int | row of the **original scan** at which this file's first row lies; 0-based, counted from the first row of the original scan. 0 for an uncropped scan |
| `scan_col_offset` | int | column of the original scan at which this file's first column lies; 0-based, counted from the first column. 0 for an uncropped scan |
| `scan_shape` | int[2] | `[rows, columns]` of the **original** scan the result was cut from (not of this file; that is `grid_shape`) |
| `eds_chemistry_prior` | int | `1` if the EDS chemistry prior (or another EDS step) decided any phase, `0` if not (≥ 1.4) |
| `phase_assignment` | str | the same fact in a sentence, with the prior's strengths and the number of pixels it changed (≥ 1.4) |
| `crystal_reference_frame` | str | the crystal frame the Euler angles refer to; see below (≥ 1.4) |

Per-pixel `/Indexing/X` and `/Indexing/Y` (µm) datasets are present from
format_version ≥ 1.2 in **both** layouts. They are the most foolproof way to
place the map on a grid in MTEX (no need to know the step or reshape by hand).
Before 1.2 the interactive *Save as… → Light .h5* path wrote neither the
coordinates nor `step_size_um`, so MTEX couldn't reconstruct the grid.

### Position in the original scan (`scan_*`)

The three `scan_*` attributes are written on `/Indexing` in both layouts and
record where this file's grid sits in the scan it was cut from (a result of
*Crop* in the viewer). They are absent from files written before they existed;
read a missing attribute as "not cropped / not recorded".

* **Axes.** *Row* is the first array axis and the sample **Y** direction
  (downwards in the map); *column* is the second axis and the sample **X**
  direction. `scan_row_offset` and `scan_col_offset` are indices (pixels), not
  micrometres, and are 0-based: `0, 0` means the file starts at the first pixel
  of the original scan. `grid_shape = [R, C]` is the size of this file; the
  window it covers in the original scan is rows `scan_row_offset …
  scan_row_offset + R − 1` and columns `scan_col_offset … scan_col_offset + C − 1`.
* **Coordinates.** `/Indexing/X` and `/Indexing/Y` (µm) are **relative to this
  file's first pixel**: they start at 0, not at the crop origin. The position
  in the original scan is
  `x_original = (scan_col_offset + column) · step_size_um` and
  `y_original = (scan_row_offset + row) · step_size_um`.
* **Matching the source.** To compare pixel for pixel with the source
  `.h5oina`, read its per-pixel arrays reshaped to `scan_shape` and take
  `[scan_row_offset : scan_row_offset + R, scan_col_offset : scan_col_offset + C]`.
  `/SourceReference` names the source file.

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
   90° fix.

   The `.ang` and `.ctf` of the same result are **not** in this frame: they hold
   the native (EDAX TSL / orix / kikuchipy) sample frame, which is what orix
   reads without conversion. For an Oxford source,
   `phi1(.h5) = phi1(.ang) − 90°` (mod 360); `Phi` and `phi2` are identical. Each
   file says so in its header.

   **Source-scan geometry is not applied.** From format 1.4 the file carries the
   source header's `Scanning Rotation Angle`, `Specimen Orientation Euler` and
   `Tilt Angle` under `/Acquisition`, under their h5oina names and in radians,
   and `/Acquisition@applied_to_euler_angles = 0` states that none of them was
   applied to `euler_angles`. The current MTEX `loadEBSD_h5.m`
   (`map_correction_scanRotation`) turns the Scanning Rotation Angle of an
   h5oina into its `EulerCorrection`, a rotation about the surface normal by
   that angle. A reader that wants the frame MTEX gives the source `.h5oina`
   has to apply that angle itself; the `.h5` Euler angles are the vendor-stored
   ones, as the source file holds them.

   **Crystal frame of non-cubic phases (format ≥ 1.4).** Every orientation
   Orienta computes — Hough, dictionary and spherical indexing alike — refers to
   the crystal frame of orix and EMsoft: **X‖a, Z‖c\*** (Y completes a
   right-handed set). Hough takes its reflectors from orix, and the dictionary
   and spherical masters are EMsoft simulations, whose Cartesian crystal frame
   is the same. MTEX's default crystal frame is **X‖a\*, Z‖c**, so in MTEX the
   `crystalSymmetry` for these orientations has to be created with
   `'X||a','Z||c*'` (see MTEX's *Crystal Reference System* page). The two
   conventions are the same frame for cubic, tetragonal and orthorhombic phases
   and differ for hexagonal, trigonal, monoclinic and triclinic ones — e.g.
   Fe₄Al₁₃ (C2/m). The statement is the same for all three routes and is stored
   in `/Indexing@crystal_reference_frame` and on every `/Indexing/Phases/<k>`.
   It was derived from the code of these libraries, not measured against a
   monoclinic reference sample. Euler angles that come from an Aztec solution
   itself are not part of any Orienta result and are never converted.

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
| 1.4 | 2026-10-08 | additions only; readers of 1.3 keep working. `/Acquisition` (source header values under their h5oina names, none applied to `euler_angles`); `/Indexing/Phases/<k>` gains `space_group` (also written when the result carries none), `space_group_symbol`, `lattice_constants` (angstrom / degrees), `lattice_length_unit`, `lattice_source`, `crystal_reference_frame`; `/Indexing` gains `eds_chemistry_prior`, `phase_assignment`, `crystal_reference_frame`. The `scan_*` attributes (already written since 1.3) are now documented. The `.ang`/`.ctf` exporters were corrected in the same change (micrometre coordinates, lattice constants, TSL symmetry codes, header statements). |

## Where this format is defined in code

* Writer (Light branch in the export route):
  [`backend/api/routes/indexing.py`](../backend/api/routes/indexing.py) — search for `elif fmt == "h5_light":`
* Writer (Light helper for batch workflow):
  [`backend/api/services/result_exporter.py::export_result_h5_light`](../backend/api/services/result_exporter.py)
* Reader:
  [`backend/api/routes/analysis.py::_load_kikuchipy_rich_h5`](../backend/api/routes/analysis.py)
* Re-import as gallery entry:
  [`backend/api/routes/indexing.py`](../backend/api/routes/indexing.py) — search for `@router.post("/import-h5")`
