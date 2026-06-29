# Phase Map

## What it does

The Phase Map page visualises an indexing result as a **spatial map** of the scan.
It renders the per-pixel phase assignment, crystal orientation (IPF colouring),
and quality metrics, and composes them into a multi-layer image you can export for
reports and publications.

It is built around a **layer stack**: each layer is one map type — phase, IPF-X/Y/Z,
band contrast (BC), confidence index (CI, overall or per-phase), uncertainty,
EDS elements, electron images, and (for spherical results) forward-NCC and
refinement diagnostics — with per-layer opacity, blend mode, visibility, and
reordering. On top of that it offers a phase legend, an IPF colour key, live map
cleanup, region/line measurement tools, an anomaly browser, and several export
paths. It is also the launch point for the detached [Pole Figure](PoleFigure.md)
window.

It uses the backend `/api/phasemap/*` routes for rendering, per-layer images, the
IPF key, legend/statistics, cleanup, probe/region/linescan measurements, and
export.

## When to use it

Use the Phase Map **after indexing** (it shows the active indexing result), to:

- See where each phase sits and how grains are oriented across the scan.
- Overlay quality (CI/BC) or chemistry (EDS) to judge and clean the result.
- Apply light cleanup (drop low-confidence pixels, remove tiny clusters, modal
  filter) before exporting or handing off to Analysis.
- Inspect individual pixels' pattern matches and browse anomalies.
- Produce a composed, annotated figure and open pole figures.

## How to use it — step by step

### 1. Choose the base map and layers

1. The page renders the **active indexing result** automatically. Pick the base
   display via the layer/display controls: **Phase Map**, **IPF-Z / IPF-X / IPF-Y**,
   **CI Heatmap**, or (spherical only) **Forward NCC**.
2. In the **Layers** panel, **+ Add Layer** to stack additional maps (IPF, BC, CI,
   per-phase CI, uncertainty, EDS elements, electron images). For each layer adjust
   **opacity**, **blend mode** (Normal/Multiply/Screen/Overlay), **visibility**, and
   drag to **reorder**. Several **presets** (Phase, Aztec-style IPF+BC, Quality
   Check, EDS Verify) configure common stacks in one click.

### 2. Read the legend and keys

3. The **Phase Legend** lists each phase, its colour (editable), and its pixel
   fraction. The **IPF colour key** shows the stereographic triangle(s) for the
   point group(s) present.

### 3. Measure and inspect

4. Use the **Tool toolbar** to probe values: hover for a multi-layer tooltip,
   drag a rectangle for **region statistics**, or draw a **line scan** to get a
   profile plot. A **magnifier lens** and **A/B swipe compare** help detailed
   inspection.
5. Click a pixel (or use the **Anomaly Browser**, which ranks suspicious pixels by
   a histogram) to open the [Pattern Match](PatternMatch.md) inspector at that
   pixel via **View Pattern Matches**.

### 4. Clean up the map

6. In the **Cleanup** group set thresholds: minimum **CI**, maximum
   **uncertainty**, minimum **cluster size**, **modal filter** size (0/3/5), and
   **fill unindexed**. These preview live on the rendered map. Click **Apply** to
   write the cleanup into the stored result (so exports and the Analysis handoff see
   it).

### 5. Diagnostics and refinement (spherical)

7. **Compute Diagnostics** builds forward-simulation diagnostic layers (Forward NCC,
   anomaly, PC sensitivity, residual) on demand, with a bandwidth choice and
   progress/cancel. **Refinement** runs the joint orientation/PC refinement and adds
   its diagnostic layers, with an original/refined view toggle.

### 6. Coordinate system, pole figures, export

8. Open the **Coordinate System** panel to set the orientation reference frame
   (rotation preset/axis-angle/Euler) and plot convention; changes update the IPF
   colours. **Open Pole Figure** launches the detached [Pole Figure](PoleFigure.md)
   window for the current result and frame.
9. In **Export**, set the **DPI** and save the rendered figure as **PNG / SVG /
   PDF**, **Copy to clipboard**, or use **Composed PNG** to bake the full layer
   stack + annotations (legend, scalebar, title, arrows) into one high-resolution
   image. **Refresh Preview** re-renders the base figure.

## Inputs & outputs

- **Inputs:**
  - The **active indexing result** (CrystalMap) in backend memory.
  - The auto-linked source H5OINA/HDF5 (for BC, EDS element maps, and electron
    images), aligned to the result's grid.
  - The chosen coordinate-system frame (shared with the Pole Figure window).
- **Outputs:**
  - On-screen layered map, legend, statistics, and measurements.
  - A **cleaned** CrystalMap when you apply cleanup (in memory; flows to exports and
    Analysis).
  - Exported images: PNG / SVG / PDF of the rendered figure, a composed PNG of the
    full stack, or a clipboard copy.

## Tips & notes

- **Per-phase CI is a within-phase measure.** A degenerate or near-degenerate phase
  can win pixels by CI alone; trust the **Forward NCC** layer (spherical) and the
  [Pattern Match](PatternMatch.md) R score for true correctness, and reduce the
  phase list before indexing if phases are physically indistinguishable.
- **Live cleanup vs. applied cleanup.** Sliders preview on the rendered map only;
  **Apply** is what changes the stored result. Nothing on disk is modified until you
  export.
- **Source linking.** EDS/BC/electron-image layers require the original file to be
  loaded and shape-aligned with the result. If shapes differ, alignment falls back
  to crop or is disabled per layer rather than mis-registering.
- **Spherical diagnostics are GPU/SHT work.** Compute Diagnostics and Refinement run
  the forward operator and can take tens of seconds; they show progress and can be
  cancelled.
- **Refinement is experimental.** The joint orientation+PC refinement effectively
  refines orientation; the pattern-centre part is a known limitation (the renderer's
  PC gradient path is incomplete), so PC deltas may be near zero. Use the
  convergence-status diagnostic layer to see where it did/didn't converge.
- **Restarting the backend clears the result.** Reloading the file re-activates its
  most recent result automatically; a fresh backend start needs re-indexing.
