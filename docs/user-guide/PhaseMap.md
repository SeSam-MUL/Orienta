# Phase Map

## What it does

The Phase Map page visualises an indexing result as a **spatial map** of the scan.
It renders the per-pixel phase assignment, crystal orientation (IPF colouring),
and quality metrics, and composes them into a multi-layer image you can export for
reports and publications.

It is built around a **layer stack**: each layer is one map type — phase, IPF-X/Y/Z,
band contrast (BC), confidence index (CI, overall or per-phase), uncertainty,
EDS elements, electron images, and (for spherical results) forward-NCC,
refinement and phase-check diagnostics — with per-layer opacity, blend mode,
visibility, and reordering. On top of that it offers a phase legend, an IPF
colour key, live map cleanup, region/line measurement tools, an anomaly browser,
and several export paths. It is also the launch point for the detached
[Pole Figure](PoleFigure.md) window.

The page is **map-first**: results appear as a slim **tab strip** at the top
(click a tab to activate a result; Rename / Delete / Save as… / Add file… /
→ Analysis sit next to the tabs), the toolbar above the canvas holds the view
tools (Stack/Grid, Linescan, Lens, **Clean view**, Export PNG, **View pattern
matches**), and the map itself takes all remaining space. Compute-heavy tools
(Forward Diagnostics, R+PC Refinement, Pseudo-Symmetry, Phase Verification)
live in a collapsed **Advanced tools** group in the sidebar.

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
   fraction. The **IPF colour key** opens as a vertical panel **beside the map**
   (toggle chip below the canvas) — one stereographic triangle per Laue class
   present.
4. On any IPF layer two extra controls appear:
   - **Phase:** — show only ONE phase as an IPF map (all others go
     transparent). Each phase has its own colour key (own fundamental sector),
     so mixed IPF maps with identical RGB codes are ambiguous — the community
     standard is one IPF map per phase. The on-screen key follows the
     selection, and the composite export captures exactly this view.
   - **Stabilize colors per grain** — display-only fix for low-symmetry
     phases (e.g. m-3 approximants) whose colour key is discontinuous: without
     it, ~1° orientation noise renders smooth grains as colour speckle.
     Gradients inside grains are preserved; stored orientations are untouched.
     See `docs/ipf-colour-maps.md` for the full story.

### 3. Measure and inspect

5. Use the **Tool toolbar** to probe values: hover for a multi-layer tooltip,
   drag a rectangle for **region statistics**, or draw a **line scan** to get a
   profile plot. A **magnifier lens** and **A/B swipe compare** help detailed
   inspection. **Clean view** hides all canvas overlays (title, scalebar) for
   presenting and screenshots.
6. Click a pixel (or use the **Anomaly Browser**, which ranks suspicious pixels by
   a histogram) to open the [Pattern Match](PatternMatch.md) inspector at that
   pixel via **View pattern matches** in the toolbar.

### 4. Clean up the map

7. In the **Cleanup** group set thresholds: minimum **CI**, maximum
   **uncertainty**, minimum **cluster size**, **modal filter** size (0/3/5), and
   **fill unindexed**. These preview live on the rendered map. Click **Apply** to
   write the cleanup into the stored result (so exports and the Analysis handoff see
   it).

### 5. Advanced tools (spherical results)

All of these live in the collapsed **Advanced tools** group in the sidebar:

8. **Compute Diagnostics** builds forward-simulation diagnostic layers (Forward NCC,
   anomaly, PC sensitivity, residual) on demand, with a bandwidth choice and
   progress/cancel. **Refinement** runs the joint orientation/PC refinement and adds
   its diagnostic layers, with an original/refined view toggle.
9. **Pseudo-Symmetry → Unify variants (whole map)** removes pseudo-symmetry
   variant speckle per grain (render-verified; coherent twin domains are
   protected; one-level undo via the Pattern-Match dialog).
10. **Phase Verification (render)** finds grains stored as the WRONG phase
    (chemically degenerate phases can win pixels of another phase because
    per-phase indexing scores are not comparable — only the forward render
    is). **Check phases** is read-only and adds a **Phase Check** layer
    (green = stored phase wins, red = another phase renders clearly better);
    **Reassign N grains** then flips only clearly-losing grains (whole grain,
    per-pixel Hough orientation, margin ≥ 0.05, one-level **Undo**).

    **The check has two stages, and the button applies both.** Stage 1 works
    on whole grains — **9 pixels or more**. Stage 2 catches what that leaves
    behind: the smaller **islands sitting inside a grain**, completely
    surrounded by another phase. Those are a different
    defect — in heavily deformed material the patterns go diffuse, the fine
    detail that identifies a low-symmetry intermetallic is the first thing to
    go, and the few broad bands that survive fit any cubic phase. The result
    is matrix pixels scattered through an intermetallic particle.

    Stage 2 needs no Hough: an island inside a grain is surrounded by pixels
    whose orientation is already known, so the candidate orientation is the
    neighbour's (taken **per pixel**, not averaged over the island — grains
    drift, and an average would be wrong exactly where it matters). It is
    therefore renders only, and cheap: measured on a deformed 7050 map
    (39×136), stage 1 found **3 grains / 23 px** while stage 2 found **19
    islands / 22 px** in **4.0 s** — 17 of them Al pixels inside an Al7FeCu2
    particle, which was the defect actually being reported. Of 30 islands
    examined it left **11 alone**; the wrong pixels rendered a median 0.229
    against 0.489 for the proposal.

    **Neither stage believes a loss at face value.** Stage 1 samples 16
    pixels of a suspect grain and, before offering a reassignment, also asks
    Hough for an orientation of the *stored* phase on those pixels — one
    batched call per phase — and judges the grain at the better of the two.
    This is what stops a low-symmetry phase from being deleted for its
    indexer's orientation: on the 7050 map the whole MgCuAl2 particle (one
    grain, ~250 px) was flipped to Al before this step existed, although at a
    fair orientation MgCuAl2 rendered 0.420 against Al's 0.411 — a tie, not a
    loss. Grains the step could not settle are reported as **undecided** and
    left alone; grains it saved are counted as **rescued**.

    **Stage 2 does not believe a loss at face value either.** An island's stored
    orientation is unreliable almost by definition — these are the pixels
    where indexing already went wrong — and a wrong orientation renders badly
    whatever the phase is. Measured on the 7050 map, a third of the raw
    stage-2 findings were exactly that: the stored phase, given a fair
    orientation, was the *better* one (MgCuAl2 rendered 0.207 at its stored
    orientation and 0.420 at a fair one). So before a finding is offered, the
    stored phase is re-oriented by Hough and judged at the better of the two.
    Findings that survive are offered; those that turn out to be a bad
    orientation are dropped (counted as *rescued*); and where Hough can find
    no orientation at all, the island is reported as **undecided** and left
    alone — the check cannot tell a wrong phase from a wrong orientation
    there, and says so rather than guessing.

    Expect *undecided* to be the common outcome, not the exception. Measured
    on the 7050 map: of 59 islands that lost at their stored orientation,
    Hough could orient the stored phase on only 4 (1 rescued, 3 confirmed);
    the other **55 came out undecided**, because Hough cannot fit bands to a
    single noisy pixel any better than the original indexing could. All 8
    findings that the orientation search had proven to be artefacts were
    stopped — by "cannot judge", not by "judged and found fine". That is the
    trade: far fewer automatic repairs, none of them of the kind that removes
    a correct phase. The undecided islands are the ones to look at by hand in
    the Pattern-Match dialog, where *Compare phases* re-indexes the pixel per
    phase with the full search the check cannot afford.

    **The grain decides the candidate; the pixel decides whether it is
    written.** When you press Reassign, every pixel of a chosen grain or
    island is verified on its own before it changes: the new phase, at that
    pixel's own Hough orientation, must render at least as well as a correct
    assignment does (0.25) *and* clearly better than the phase it replaces.
    Pixels that fail stay as they are and are reported as "left unchanged —
    too little evidence". Measured on the 7050 map: a single Reassign had
    moved 531 pixels; per pixel, the ones that pass this test have the band
    contrast of the particle (110), the ones that fail have that of the
    matrix (83), and 44 pixels handed to MgCuAl2 rendered at 0.09 — worse
    than the phase they replaced. A grain median of 16 samples cannot tell
    those apart; the pixel can. Pixels for which Hough finds no orientation
    of their own are never filled from a neighbour any more.

    The two counts are always reported apart ("3 grains + 19 pixels"), never
    summed: they are different repairs, and one number you cannot take apart
    hides which one happened. Both share the single **Undo** — note that the
    undo slot is one for all phase edits, so a manual assignment made after a
    Reassign replaces the Reassign's undo.

    *Current limitation:* stage 1 runs synchronously and can take minutes on
    large multi-phase maps — a progress/cancel version is planned. Stage 2 is
    not the slow part. Stage 2 also only sees islands **completely enclosed by
    a single other phase**: a wrong patch on a particle's edge, or one bordering
    two different phases, is still missed, because there is then no reliable
    candidate orientation to be had without Hough.

    **If "Assign … to this grain" reports that Hough failed.** The manual
    assignment in the Pattern-Match dialog normally gives each pixel its own
    Hough orientation for the target phase. When Hough cannot run at all it now
    falls back to the orientation the dialog already computed and showed you
    (the one whose R you just read), carried across the grain as a rigid
    correction — so the grain keeps its internal misorientation and only its
    anchor changes. That is the weaker of the two sources, so the confirmation
    message says when it was used.

    A failure here is often reported by the driver as
    `Context failed: OUT_OF_HOST_MEMORY`, which sounds like a memory shortage
    and usually is not: PyEBSDIndex runs its Radon transform through OpenCL,
    and this is that context failing to open. It hits every phase equally —
    observed on a plain `Al` CIF (m-3m, 50 reflector families, a *negligible*
    library) with 6.7 GiB free. The app therefore checks what the library would
    actually cost against what is free before it blames memory, and prints both
    numbers so you can see which case you are in.

### 6. Coordinate system, pole figures, export

11. Open the **Coordinate System** panel to set the orientation reference frame
    (rotation preset/axis-angle/Euler) and plot convention; changes update the IPF
    colours. **Open Pole Figure** launches the detached [Pole Figure](PoleFigure.md)
    window for the current result and frame.
12. In **Export**, set the **DPI** and save the rendered figure as **PNG / SVG /
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

- **Layers follow a crop.** If the active dataset is a cut-out from the EBSD
  Viewer (see `EBSDViewer.md`), the EDS element maps, Band Contrast and the phase
  and IPF layers all show the same region. An electron image that cannot be
  placed from the file's own geometry is shown whole with a warning chip in its
  layer row, rather than being cut to a guessed position.
- **Per-phase CI is a within-phase measure.** A degenerate or near-degenerate phase
  can win pixels by CI alone; trust the **Forward NCC** layer (spherical) and the
  [Pattern Match](PatternMatch.md) R score for true correctness, and reduce the
  phase list before indexing if phases are physically indistinguishable. The
  **Phase Verification** tool automates exactly this render-based cross-check.
- **Live cleanup vs. applied cleanup.** Sliders preview on the rendered map only;
  **Apply** is what changes the stored result. Nothing on disk is modified until you
  export.
- **Careful with the CI-threshold slider on spherical results.** Spherical "CI"
  values are correlation peaks that typically sit around 0.3–0.4 even for
  perfectly indexed pixels — a threshold of 0.3 can grey out half a good map.
  Keep it at 0 for spherical results unless you are deliberately filtering;
  use Forward NCC for genuine quality filtering.
- **Loading files.** Use the tab strip's **Add file…** — it accepts `.ang` /
  `.ctf` (orientations only), GUI-exported rich/light `.h5` (full result,
  optionally with patterns/EDS wired back in), and `.npy` phase arrays.
  Re-importing a rich/light `.h5` restores the **complete session**: the
  indexing method, experimental patterns (Oxford and EDAX layouts), the
  detector geometry, and the per-phase `.sht` simulations — matched via the
  provenance stored in newer exports, or by phase name / element-ratio
  against your SHT library for older files — so Pattern Match, Compare
  phases and the grain-flip tools work exactly as on a fresh run. Results
  saved from a region-of-interest come back as that ROI (maps and the NCC
  heatmap crop to it).
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
  most recent result automatically; a fresh backend start needs re-indexing — or
  simply **Save as… → Rich .h5** before the restart and **Add file…** it back
  afterwards: the full session (patterns, simulations, edits) is restored.
