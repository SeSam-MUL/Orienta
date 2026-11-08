# Analysis

## What it does

The Analysis module is the post-processing stage for an indexed EBSD scan — a
Python equivalent of a typical MTEX workflow. Starting from a crystal-orientation
map (one orientation + quality value per pixel), it:

- **reconstructs grains** by grouping neighbouring pixels whose misorientation is
  below a threshold,
- computes **grain-size statistics** (equivalent circle diameter, max dimensions,
  aspect ratio),
- measures **local deformation** — KAM (Kernel Average Misorientation), grain
  boundaries split into low-angle (SAGB) and high-angle (HAGB), and a 3-Gaussian
  Band-Contrast model that separates deformed / recovered / recrystallized
  fractions,
- analyses **crystallographic texture** (texture index, entropy, named texture
  components, surface-plane fractions), and
- classifies **recrystallization** (RX) from GOS / band-contrast / grain-KAM
  criteria.

It renders these as colour maps (BC, IPF, KAM, GOS, ECD, RX, grain boundaries)
and can export every result to a multi-sheet Excel workbook. The heavy maths
lives in the backend `analysis/` package; the page drives it through the
`/api/analysis/*` routes.

## When to use it

Use Analysis **after indexing** (or after running a batch), once you have an
orientation map. It is where you turn that map into quantitative microstructure
metrics:

- characterise grain size and shape for a sample,
- quantify stored deformation and the high-/low-angle boundary network,
- describe the texture (which orientation components dominate), and
- estimate how much of the structure is recrystallized.

The six tabs follow the natural workflow order: **Data & Preprocessing → Grain
Analysis → Deformation → Texture → Recrystallization → Batch & Export**.

## How to use it — step by step

### Tab 1 — Data & Preprocessing

1. Load an orientation map. The easiest path: pick the **session result** from the
   "Session Results" dropdown (your most recent indexing run). Alternatively, type
   or **Browse** to a map file (`.h5`, `.hdf5`, `.h5oina`, `.ang`, `.ctf`) and
   click **Load**. (Batch and other modules can also hand a result here directly.)
2. Confirm the dataset info that appears: grid shape, phase list, and step size.
3. *(Optional)* If the loaded map carries Band Contrast / Bands quality fields, a
   **Quality Filter** box appears. Set **Min Band Contrast** / **Min Bands** and
   apply it to mark low-quality pixels as unindexed so they are skipped by grain
   reconstruction, KAM/GOS, texture, and export. This is destructive — reload to
   reset. Grains that were reconstructed before the filter are not changed:
   reconstruct them again.
4. *(Optional)* **Aztec Comparison** renders a 3-panel figure comparing Aztec's
   stored phase + orientation against Orienta's (phase agreement %, misorientation
   stats), when the original H5OINA can be located next to the result.

### Tab 2 — Grain Analysis

5. Set the **GB Threshold** (the misorientation threshold in degrees, default 4° on
   the page, more than 0) and **Min Grain Size** (pixels, a whole number of 1 or
   more). Click **Reconstruct Grains**. Reconstruction runs in the
   background; a progress indicator polls until done and reports the grain count
   plus mean/median ECD and aspect ratio. **Min Intercept** is only recorded in
   the workbook (Sheet1, row 11); the reconstruction does not use it, and an
   empty field leaves the workbook cell empty. Two further settings control how
   gaps in the map are treated:
   - **Close holes (alpha)** (default 3.1, 0 to 20). Unindexed pixels inside a grain, such as holes,
     scratches or a thin unindexed band, are closed when they are narrower than
     about 2 × alpha pixels, so the grain stays in one piece and its outline
     follows the real grain edge. Larger values close wider gaps; **0 closes
     nothing**, and every unindexed pixel then stays a gap. The method follows
     Bachmann, Hielscher and Schaeben, *Ultramicroscopy* 111 (2011) 1720–1733,
     doi:10.1016/j.ultramic.2011.08.002.
   - **Fill holes and scratches in maps** (default on). Shows the closed pixels with
     the orientation of the grain they lie in, so IPF maps look continuous. This is
     **display only**: filled pixels are flagged, and the orientation given to them
     is never used in a number. Grain sizes (ECD from the cell area, max dimensions,
     area) are measured on the grain including the closed pixels, because they
     describe the grain's extent; orientation values (GOS, KAM, mean orientation)
     are computed from measured pixels only. With the option off the closed pixels
     stay grey in the IPF map; the grains and all numbers are the same.

   Grains never span two phases: neighbouring pixels of different phases always
   belong to separate grains, whatever their misorientation. The **aspect ratio**
   is the length of the long principal axis of the grain's shape divided by
   the short one, so it is always 1 or larger.

   **Grain-size statistics.** Three settings under the fields above decide what
   goes into the means, and they are recomputed as soon as you change one (the
   workbook records them in Sheet1, rows 49 to 59):
   - **Edge grains.** Which grains are left out because they touch the border of
     the map. *Like the MATLAB reference scripts in `matlab_testskripts/`* (the default; "the script" below means these) leaves out grains with a
     measured pixel in the outermost two pixel rows or columns of the measured
     area. *Like MTEX* leaves out grains that touch the map border or a
     not-indexed area that is open to the border. *Outermost pixel row* leaves out
     grains with a pixel in the first or last row or column. *Keep all grains*
     leaves none out. The first two keep all grains instead when fewer than 50
     grains, or less than 15 % of the grains, would remain; the outermost-row rule
     does so when it would leave out more than 85 %. The page says so when it
     happens.
   - **ECD from.** *Cell area* (the default, as MTEX) takes the area of the
     measurement cells of the grain, closed holes included. *Measured pixels ×
     step²* (as the MATLAB script) counts the measured pixels only.
   - **Phase.** *All phases* (the default), one phase of the map, or, in the batch,
     *the phase with the most measured pixels* of each file. With a phase chosen
     the statistics use the grains of that phase only; with the script's edge rule
     the border strip is then drawn around that phase's own extent.

   The statistics block names the number of grains in the statistics (after the
   edge rule), the rule and the basis that were used, and the step size.
6. View the result maps (BC / IPF / ECD / grain boundaries) in the per-tab map
   viewer; switch layer and colormap with its dropdowns.

### Tab 3 — Deformation

7. With grains reconstructed, set the parameters and click **Analyze Deformation**.
   It produces the **KAM** map plus mean KAM, total grain-boundary length split
   into **HAGB** and **SAGB**, and a sphericity metric.
   - **KAM order** (a whole number from 1 to 5, default 2): the KAM of a pixel is
     its mean misorientation to the neighbours up to this many pixel steps away
     (the row and column distances add up to at most the order).
   - **KAM threshold** (degrees, more than 0 and at most 30, default 7°):
     misorientations above it are left out of the KAM.
   - **HAGB from** (degrees, more than 3 and at most 90, default 15°): boundaries
     from this misorientation up are high-angle (HAGB); **SAGB** runs from 3° to
     this limit.

   The values are sent with the run and remembered. The KAM map, the numbers, the
   boundary lengths, the Excel export, the batch and the KAM that the RX analysis
   uses follow them, and the page prints the values the numbers were made with.
   The workbook records the KAM order and threshold on the `KAM hist` sheet
   (cells B1 and B2) and the HAGB limit in Sheet1, row 71 (`hagb_min_deg`). A value
   you have typed and not run yet is marked "Not applied to this map yet".

### Tab 4 — Texture

8. Choose a **texture preset** (e.g. FCC_Rolling), the **phase** and run the
   texture analysis. The results table lists the named texture components with
   their volume fractions, plus **Texture Index**, **Entropy**, and the
   **surface-plane fractions** ({111}/{100}/{110}).

   **Phase.** The texture is computed on one phase, as in the MATLAB reference
   script (`ph = 'aluminium'`, the matrix). The default is the **phase with the
   most indexed pixels** of the map (the lower phase number on a tie); you can
   choose any other phase of the map. Every number on the page, in the workbook
   and in the batch (component fractions, the rolled-versus-recrystallized split,
   texture index and entropy, surface planes, and the **Texture Components**
   map layer) uses the pixels of that phase only, with that phase's symmetry. The result names the phase
   ("Texture of: Al"), and Sheet 1 of the workbook records it in row 67
   (`texture_phase`). The texture components are cubic (FCC and BCC presets), so
   the phase must be cubic; for a phase that is not, Orienta refuses and names the
   cubic phases of the map. In a batch, each file uses its own phase with the most
   indexed pixels.

   **Preset and tolerance angle.** The preset (FCC or BCC rolling) and the tolerance angle (more than 0° and at
   most 30°; a pixel counts for a component when it is within this angle of it) are sent with the run and
   remembered with the phase. The component fractions, the component map, the rolled-versus-recrystallized
   split, the workbook and the batch all use them (the batch takes them from the Texture tab and says so on its
   tab); Sheet 1 records them in rows 68 (`texture_preset`) and 69 (`texture_tolerance_deg`).

   **Components.** A component is {hkl}<uvw> as written in the preset, with the
   rolled-sheet symmetry (mmm) of the sample; its fraction adds the share of
   pixels within the tolerance of each of its distinct variants (1, 2 or 4), so a
   pixel that is near two variants counts twice, as in the MATLAB reference script.
   The component fractions also overlap each other: a pixel near Cube is also
   near Rot.Cube ND10 and Rot.Cube RD10, so the three columns of a pure cube
   texture add up to far more than 100 %. Do not add the columns up and do not
   read their sum as a share of the map.

   **Texture index and entropy.** These are the script's `norm(odf)^2` and
   `entropy(odf)`, computed from a kernel density of the orientations of the
   texture phase: a de la Vallée Poussin kernel with a half-width of 5°, with the
   crystal symmetry of the phase. A texture without any preferred orientation has
   a texture index of 1 and an entropy of 0; a sharper texture has a larger index
   and a more negative entropy. They do not depend on the
   chosen tolerance angle or on the components. An entropy of exactly 0 is a
   value and is written as 0 in the workbook; a missing value (a phase without a
   single orientation) leaves the cell empty. Computing them can take a minute
   or more for a map of a million pixels, depending on the computer.

   **Surface planes.** For each of {111}, {100} and {110} the page shows two
   numbers. The main one is the share of pixels for which **any plane of the
   family** is in the surface: the share of pixels whose plane normal lies
   within 15° of the sample normal (ND) for at least one plane of the family.
   Beside it is **the MATLAB
   script's number**: the script measures one direction per pixel, the
   direction (1 1 1), (1 0 0) or (1 1 0) of the orientation as the file stores
   it, so that number depends on which of the symmetric equivalents the file
   stores; on AZtec data {100} and {110} come out near 0 %. Sheet 1 has both: rows 45
   to 47 hold the script's numbers and rows 64 to 66 the family values.

   **Known difference from the script.** The script centres the sample frame
   (MTEX `centerSpecimen`) when a map has more than 400 grains and a rotation
   below 7°. Orienta does not centre, so on a map that the script centres the
   texture numbers can differ.

### Tab 5 — Recrystallization

9. Choose the **Phase** and set the RX criteria — **GOS threshold**, **gBC fraction**,
   **gKAM threshold**, **Min grain radius** — and **Classify**. You get the RX area
   fraction, the high-BC fraction, and the RX-vs-total grain count, plus the RX
   overlay map. (Grains must be reconstructed first.)

   **Phase.** As in the MATLAB reference script, which analyses one phase, the RX
   analysis counts the grains of one phase: by default the phase with the most
   indexed pixels of the map, or any phase of the map you choose. The
   classification, the RX fraction and its denominator use the grains of that phase
   only; grains of other phases are not recrystallized and are drawn in a neutral
   colour on the map. The result names the phase, and Sheet1 of the workbook
   records it in row 70 (`rx_phase`). The rolled-versus-recrystallized comparison of the Excel
   workbook (sheet `SmallVSLargeGrain`) needs this phase to be the texture phase as well. In a batch each file uses its own phase with the most indexed
   pixels. The grain-average KAM of the criteria is computed with the KAM order and
   the KAM threshold of the Deformation tab (the script uses order 2 and 5° for
   its gKAM).

   On the map, **red** marks exactly the grains
   that count in these numbers and in the workbook: recrystallized by all four
   criteria. The other grains are coloured by their GOS band (light blue: GOS below
   1.5°, yellow: 1.5° to 7.5°, brown: from 7.5°) as context, and pixels without a
   grain are grey. Not-indexed pixels that a grain encloses belong to that grain and
   take its colour. The RX criteria need **native band contrast** in the map; without
   it the page shows "needs band contrast in the map" instead of an RX fraction, and
   the map has no red.
   The four values and the phase are sent when you press **Classify** (GOS 0 to 20°, gBC 0 to 1 of the
   largest gBC, gKAM 0 to 10°, radius 0 to 100 µm; the script's values 1.25°, 0.7, 0.55°
   and 0.4 µm are the start). The page prints the criteria used next to the numbers, and
   the RX map, the Excel export (Sheet1 rows 60 to 63) and the batch use the same ones;
   they start again at the script's values when another map is loaded. The high-BC,
   low-gKAM and low-KAM fractions are the script's fixed descriptive values (0.7 × the
   largest BC, 0.55°) and do not follow the criteria.

### Tab 6 — Batch & Export

10. **Download Excel** exports all computed results for the current dataset to an
    `.xlsx` workbook (11 sheets, see "The workbook" below: basic info, grain-size / KAM / GOS / BC
    histograms, GMM fit, boundaries, sphericity, texture, RX). In the browser it triggers a
    download; in the desktop app you can pick an output path. A choice made on the
    Deformation, Texture or Recrystallization tab and not run yet is applied first, so
    the workbook follows what the tabs show; the page says which settings it applied.
11. **Batch Processing** runs the whole pipeline over a **folder** of result files
    and writes a single `Documentation.xlsx` summarising each file as a column.
    Pick the input folder and output file, then **Process Folder**; a progress bar
    shows how far it is, with the file it is at, and a log keeps the start and the
    summary. The batch takes the grain reconstruction from the Grain tab, the
    Deformation parameters, the RX criteria and the Texture tab's preset and
    tolerance as they are set there, and says so on the tab; the edge rule, the
    ECD basis and the phase of the grain-size statistics are the Batch tab's own
    (they start equal to the Grain tab's). The texture phase and the RX phase are
    the phase with the most indexed pixels of each file.

### Run everything at once

12. The persistent bottom bar's **Run Complete Analysis** button chains Grain
    Reconstruction → Deformation → Texture → RX in sequence, with the settings as
    they are on the tabs now (the Grain tab's parameters and statistics settings,
    the Deformation parameters, the Texture tab's preset, tolerance and phase, the
    RX criteria and phase), updating the per-tab "done" badges as it goes. It
    stops before it starts when one of them is not a usable number.

## The workbook (`Documentation.xlsx`)

Both exports (Download Excel and Batch Processing) write the same workbook: 11 sheets, **one column per
dataset** unless the table says otherwise. Where the dataset name stands: in row 1 of each sheet, except `KAM hist`
(row 3; rows 1 and 2 hold the KAM order and the threshold), `Grain Boundaries` (row 1 for the scalar rows and
again in row 13 above the boundary-length bins), `BC fit (3G)` and `SmallVSLargeGrain` (one row per dataset, the
name in column A). The sheets, with the unit of each number:

| Sheet | What is on it | Unit |
|-------|---------------|------|
| `Sheet1` | basic info, grain-size metrics, KAM, boundary lengths, texture index and components, RX metrics, surface planes, grain settings and counts | as labelled in column A |
| `BC hist` | band-contrast histogram, bins of 10 from 0 to 230 | share of the pixels with a band contrast, as MATLAB's `histcounts` with `probability` (adds up to less than 1 when band contrast exceeds 230) |
| `BC fit (3G)` | 3-Gaussian fit; **one row per dataset** | weights, means, widths |
| `BandContrast Group` | centre and area fraction of the high, mid and low group | BC units, fraction |
| `Grain Size hist` | equivalent circle diameter (ECD) of the grains in bins of 5 µm from 0 to 140 µm | number of grains |
| `KAM hist` | KAM in bins of 0.25° from 0 to 30° (the largest KAM threshold allowed) | probability (adds up to 1) |
| `GOS hist, area weighted` | GOS in bins of 0.25° from 0 to 15° | **percent of the total grain area** |
| `gKAM - area weighted` | grain-average KAM in bins of 0.2° from 0 to 5° | **percent of the total grain area** |
| `Grain Boundaries` | boundary lengths and densities (rows 3–10), mean sphericity (row 11), length of boundary per misorientation bin of 2° (from row 14) | µm, 1/µm, µm per bin |
| `Sphericity per grain` | the sphericity of every grain, grain number in column A | dimensionless |
| `SmallVSLargeGrain` | texture-component volume fractions of all grains, of the substructure grains and of the RX grains; **one row per dataset** (the name in column A, from row 3) | percent of the pixels of each group |

Notes:

- **Bin centres are fixed.** Each histogram sheet has one set of bin centres in column A; a dataset only writes its
  counts. Values outside the range (an ECD above 140 µm, for example) are left out, and Orienta notes how many
  in its log file (the diagnostics export under Settings contains it).
- **Percent, not pixels.** The GOS and gKAM sheets give percent of the total grain area (closed holes included),
  as MTEX's grain histogram does (it does not depend on the step size). Each grain counts with its area, the
  cells it covers, including the not-indexed pixels it encloses and closes; the total is the area of all
  grains. A grain whose value is outside the sheet's range stays in the total, so the column can add up to
  less than 100.
- **A file name that exists is never written into.** If the chosen workbook already exists, the export goes to
  `<name>_2.xlsx` (then `_3`, ...); the result names the file that was written. An empty file that a crashed run
  left behind is not a workbook: it is used again. If an export stops half-way, the unfinished file is kept as
  `<name>.incomplete.xlsx` and the message names it. A workbook from an older version of Orienta is not opened
  either: it is a file that exists, so the export goes to the next free name.
- **Differences from the MATLAB `Documentation.xlsx`.** The numbers are the same quantities; the layout is not
  always the same:
  - `Sheet1` has one column per dataset, the quantities in rows; the script writes one row per dataset. The
    texture components are in the order of the Orienta preset (Brass, S, Copper, Goss, Inv. Goss, Rot.Cube ND45,
    ND22, ND10, RD10, Cube, Q, P, R, Shear 2); the script's columns run Brass, S, Copper, Goss, Rot. Cube ND45,
    ND22, RD10, ND10, Cube, Q, P, R, shear 2 and Rot. Goss (the same orientation as Inv. Goss).
  - **One grain reconstruction.** Orienta reconstructs the grains once, on the Grain tab (or in the batch), and uses
    that reconstruction for the grain size, the RX analysis and the rolled-versus-recrystallized split. The scripts
    run `calcGrains` again in each step: the grain size with 5° and a minimum of 3 pixels on the filled data of one
    phase, the RX analysis with alpha 2, angles of 3° and 1° and smoothed data, and the SmallVSLargeGrain split with
    a GOS limit of 1° on a reconstruction at 6°. The RX numbers (and the split) can therefore differ from the
    script's for the same map. The grain-average KAM of the RX analysis uses the KAM settings of the Deformation
    tab (the script: gKAM with order 2 and 5°).
  - `Grain Size hist` has one block (ECD, counts). The script's maxDimX, maxDimY and area-weighted ECD blocks, and
    its probability normalisation, are not written.
  - `Grain Boundaries`: the first dataset is in column B (the script uses column C, with bin edges in B); the
    rows are called SAGB and "per area" where the script says LAGB and "per field area"; the segment bins are
    labelled by their centre (4, 6, ..., 64), the script's by their lower edge (3, 5, ..., 63); the sphericity
    of each grain is on its own sheet and row 11 holds its mean.
  - `KAM hist`: the script writes the bin edges under "BinCenters"; Orienta writes the bin centres.
  - `gKAM - area weighted`: the script writes the GOS bin centres here; Orienta writes the centres of
    the gKAM bins (0.1, 0.3, ...).
  - `SmallVSLargeGrain`: filled as in the script (one row per dataset, three blocks), by both exports; the
    RX/substructure split is the one of the Recrystallization tab, not the script's own GOS limit of 1°. The split
    needs the **RX phase and the texture phase to be the same phase**: the RX grains are grains of the RX phase,
    while the texture is counted on the pixels of the texture phase. When they differ, the sheet is left out and
    the export and the batch say so in their warnings. When the RX analysis is not run (the map has no native
    band contrast), the sheet has only the block of all grains.

## Inputs & outputs

- **Inputs:**
  - An orientation/crystal map: the last in-session indexing result, or a file
    (`.ang`, `.ctf`, or an Orienta-exported `.h5` / `_light.h5`). Rich/light `.h5`
    exports also carry per-pixel quality fields (BC, Bands, PC) used by the quality
    filter and PC-drift views.
- **In-session outputs:**
  - A reconstructed **grain set** and all derived maps/statistics, held in backend
    memory (and stashed per-file so they survive switching files and back).
- **File outputs:**
  - A per-dataset **Excel workbook** (single-file export or browser download).
  - A folder-level **`Documentation.xlsx`** from Batch Processing.
  - The Aztec-comparison and PC-drift figures are rendered as images in the page.

## Tips & notes

- **Band-Contrast analyses need *native* Band Contrast.** The 3-Gaussian
  Band-Contrast model (Deformed / Recovered / Recrystallized) in Deformation, the
  **Quality Filter** in Tab 1 and the **RX analysis** (its band-contrast
  criterion) all require the vendor's real per-pixel Band
  Contrast — i.e. an Oxford H5OINA (or a light-`.h5` that carried it forward). On
  datasets without it (EDAX, synthetic data, some indexing results, where only a
  computed FFT "Pattern Quality" is available) the first two fail loudly with a clear
  message rather than silently running the model on a surrogate, and the RX
  analysis gives no RX fraction and the RX map no red. Grain reconstruction, KAM,
  GOS and texture work regardless.
- **Reconstruct grains first** for the GOS, ECD, RX and grain-boundary maps, the
  deformation analysis and the RX analysis: the page asks for the reconstruction
  instead of drawing them. The texture needs no grains; it is computed from the
  pixels of the chosen phase.
- **Need a real 2-D map.** Maps and statistics require a 2-D scan; a single-pixel
  Quick-Test result has no map and is rejected with an explanatory error.
- **Step size matters for absolute sizes.** ECD and boundary lengths are in µm and
  depend on the step size. Orienta reads it from the file when present and falls
  back to the active EBSD signal otherwise; verify the step shown in Tab 1.
- **Crop, rotate and smooth preprocessing is not available.** The Data tab does
  not offer it. Use the Quality Filter (when available) for the cleanup that is
  implemented.
- **Results live in memory.** Restarting the backend clears the loaded map and
  grains; reload the result to continue.
