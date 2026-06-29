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
   reset.
4. *(Optional)* **Aztec Comparison** renders a 3-panel figure comparing Aztec's
   stored phase + orientation against Orienta's (phase agreement %, misorientation
   stats), when the original H5OINA can be located next to the result.

### Tab 2 — Grain Analysis

5. Set the **Misorientation threshold** (degrees, default 5°) and **Min grain
   size** (pixels). Click **Reconstruct Grains**. Reconstruction runs in the
   background; a progress indicator polls until done and reports the grain count
   plus mean/median ECD and aspect ratio.
6. View the result maps (BC / IPF / ECD / grain boundaries) in the per-tab map
   viewer; switch layer and colormap with its dropdowns.

### Tab 3 — Deformation

7. With grains reconstructed, click to run **Deformation** analysis. It produces
   the **KAM** map plus mean KAM, total grain-boundary length split into **HAGB**
   and **SAGB**, and a sphericity metric.

### Tab 4 — Texture

8. Choose a **texture preset** (e.g. FCC_Rolling) and run the texture analysis.
   The results table lists the named texture components with their volume
   fractions, plus **Texture Index**, **Entropy**, and **surface-plane fractions**
   ({111}/{100}/{110}).

### Tab 5 — Recrystallization

9. Set the RX criteria — **GOS threshold**, **gBC fraction**, **gKAM threshold**,
   **Min grain radius** — and **Classify**. You get the RX area fraction, the
   high-BC fraction, and the RX-vs-total grain count, plus the RX overlay map.
   (Grains must be reconstructed first.)

### Tab 6 — Batch & Export

10. **Download Excel** exports all computed results for the current dataset to an
    `.xlsx` workbook (multiple sheets: basic info, grain-size / KAM / GOS / BC
    histograms, GMM fit, boundaries, texture, RX). In the browser it triggers a
    download; in the desktop app you can pick an output path.
11. **Batch Processing** runs the whole pipeline over a **folder** of result files
    and writes a single `Documentation.xlsx` summarising each file as a column.
    Pick the input folder and output file, then **Process Folder**; a progress bar
    and log track it.

### Run everything at once

12. The persistent bottom bar's **Run Complete Analysis** button chains Grain
    Reconstruction → Deformation → Texture → RX in sequence (with default
    parameters), updating the per-tab "done" badges as it goes.

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

- **Reconstruct grains first.** GOS, ECD, RX, grain-boundary maps, deformation,
  and texture all require a grain set — running them before reconstruction returns
  a clear "reconstruct grains first" message.
- **Need a real 2-D map.** Maps and statistics require a 2-D scan; a single-pixel
  Quick-Test result has no map and is rejected with an explanatory error.
- **Step size matters for absolute sizes.** ECD and boundary lengths are in µm and
  depend on the step size. Orienta reads it from the file when present and falls
  back to the active EBSD signal otherwise; verify the step shown in Tab 1.
- **The "Apply Preprocessing" button in Tab 1 is a placeholder.** Crop/rotate/
  smooth preprocessing is not yet wired up — it reports "not implemented". Use the
  Quality Filter (when available) for the cleanup that is implemented.
- **Results live in memory.** Restarting the backend clears the loaded map and
  grains; reload the result to continue.
