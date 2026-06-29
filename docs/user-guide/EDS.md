# EDS

## What it does

The EDS (Energy-Dispersive X-ray Spectroscopy) module turns the per-pixel
element counts stored inside an Oxford H5OINA scan into a chemistry view of your
sample. It reads each element's count map, converts counts to **Weight %** and
**Atomic %** on the fly (Cliff–Lorimer-style normalisation), and presents the
result as:

- a curated **composite overlay** (a small stack of layers — phase, IPF, band
  contrast, chosen elements — blended together), and
- an **All Maps** tile grid that shows every detected element map, every electron
  image, and band contrast at once.

On top of the maps it offers per-pixel and per-region **quantification**,
chemistry-driven **phase suggestion**, and an **auto-classify phase-map builder**
that paints a phase map from the local chemistry. The element-to-chemistry maths
runs in the backend `eds_utils` module; all map, probe, quantify, and phase-map
operations go through the `/api/eds/*` routes.

## When to use it

Use EDS after loading a scan with EDS data (in the EBSD Viewer), and typically
**before indexing**, to:

- See where each element sits across the scan and identify chemically distinct
  regions (matrix vs. intermetallics, precipitates, inclusions).
- Read the exact composition (At.% / Wt.% / counts) at a pixel or averaged over a
  region you draw.
- Get a short list of **candidate phases** that match the measured chemistry, so
  you index against plausible phases rather than guessing.
- Build a quick **chemistry-based phase map** and hand its phase list and pixel
  masks to the Indexing page for chemistry-guided, phase-selective indexing.

EDS quantification is semi-quantitative (no per-element standards or full ZAF
correction) — treat it as a guide for phase selection and regional comparison,
not as a certified composition measurement.

## How to use it — step by step

### Open and choose a display mode

1. Load an H5OINA scan that contains EDS data in the **EBSD Viewer** first. If no
   file is open, the EDS page shows an empty-state prompt.
2. In the header, pick the **display mode** with the toggle: **Counts**, **Wt.%**,
   or **At.%**. This drives every element layer, the tooltip, and quantification.
   At.% is the default and is the correct basis for phase matching.
3. The header **File switcher** lets you switch between several scans loaded this
   session.

### Read the maps

4. The **Composite Overlay** (left) shows the curated layer stack. Use the
   **Layers** panel below it to add/remove layers (**+ Add Layer**), toggle
   visibility, change opacity and blend mode, and reorder them by dragging.
   Quick-mode buttons (Phase / IPF-Z / BC / CI) replace the stack with a single
   layer. Available "add" options include any indexing result layers (Phase, IPF,
   CI), every electron image, and every EDS element.
5. The **All Maps** tile grid (centre) shows every element map, electron image,
   and band contrast simultaneously. Use the **Tile size** slider to resize the
   tiles.
6. **Hover** any map to get a tooltip with all element values, band contrast, and
   (if a phase map exists) the phase at that pixel, in one lookup. Toggle the
   **Lens** (4× magnifier), the **Linescan** tool (drag a line to plot per-layer
   profiles), and **Export PNG** to save the composite. A **swipe compare**
   splitter lets you wipe between two layers.

### Quantify

7. In the right rail's **Pixel Quantification** box, enter a **Row** and **Col**
   (or click a pixel on any map), then **Quantify** to get the Element / Counts /
   Wt.% / At.% table. The footer shows the At.% sum with a ✓/⚠ check that it
   normalises near 100 %. **Copy** puts the table on the clipboard as TSV.
8. In **Region Average**, type a rectangle (Row/Col start–end) or **Shift-drag** a
   box on a map; the page fills the fields and computes the mean ± standard
   deviation per element over the region.

### Build a phase map (optional)

9. In the **Phase Map** controls, set the **Tolerance** (allowed per-element At.%
   deviation) and **Min Score** sliders, then click **Auto-Classify**. Every pixel
   is matched against your curated CIF library (`Database/crystal_database.xlsx`)
   and the best-scoring phase is assigned; a colour legend with per-phase area
   fractions appears.
10. Refine the map manually: pick a phase in the legend, choose **Rectangle** or
    **Polygon** paint mode, draw on the map, and **Assign** to overwrite those
    pixels with that phase. Use phase `-1` to mark pixels unclassified.
11. Use the **send-to-indexing** action to hand the phase map's CIF filenames and
    pixel masks to the Indexing page for chemistry-guided indexing.

### Suggest phases

12. In **Phase Suggestion**, set Row/Col (or click a pixel) and **Suggest**. The
    chemistry at that pixel is matched against your CIF library (or a built-in
    fallback library if you have not built one yet); each candidate shows formula,
    space group, crystal system, and a match score.

## Inputs & outputs

- **Inputs:**
  - An open H5OINA scan containing per-element EDS count maps (`/1/EDS/Data/Window
    Integral/<element>`).
  - Optionally your curated CIF library
    (`Database/crystal_database.xlsx`) for phase suggestion and auto-classify;
    optionally an indexing result for Phase/IPF/CI overlay layers.
- **In-session outputs:**
  - Quantification tables (per-pixel and per-region), copyable as TSV.
  - A stored **phase map** (held per-process) that other tools and the
    send-to-indexing hand-off can consume.
  - A combined and per-phase **pixel mask** + CIF filename list forwarded to
    Indexing.
- **File outputs:** the composite overlay exported as a **PNG**. Quantification is
  copied to the clipboard rather than written to disk.

## Tips & notes

- **Match phases on At.%, not counts.** Counts are raw detector readings; the
  At.% conversion is what makes chemistry comparable to a crystal formula. The UI
  defaults to At.% for exactly this reason.
- **Semi-quantitative.** The conversion is a simplified normalisation, not a
  standards-based ZAF quantification. Use it for relative comparison and phase
  shortlisting.
- **Auto-classify needs a CIF library.** If `Database/crystal_database.xlsx` is
  missing or contains no phase whose elements are a subset of the measured
  elements, auto-classify reports an error — build/curate the database first
  (see the Crystal Database module).
- **Aztec's pre-rendered RGB images are excluded by design.** The pre-indexed
  EBSD/EDS "layered images" inside the H5OINA are intentionally not loaded, since
  Orienta does its own indexing.
- **The hover tooltip is deliberately limited.** It returns element values, band
  contrast, and phase only — electron-image and virtual-BSE values are omitted to
  keep the lookup within the tooltip's response budget.
- **The phase map lives in memory.** It is held per backend process; restarting
  the backend clears it, and re-running Auto-Classify replaces any manual paint
  edits.
