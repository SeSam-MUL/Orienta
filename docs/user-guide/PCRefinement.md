# PC Refinement

## What it does

PC Refinement calibrates the **pattern centre (PC)** and detector geometry for an
EBSD scan. The pattern centre — the projection of the sample point onto the
detector, expressed as `(PCx, PCy, PCz)` — together with the sample and detector
tilts, determines how Kikuchi bands map onto the detector. Accurate geometry is
the single most important prerequisite for correct indexing: if the PC is wrong,
indexing is wrong.

The page lets you:

- Load one or more crystal **phases** (from CIFs) to provide reflectors for
  indexing — for example austenite and ferrite of a duplex steel.
- **Index** the calibration patterns you collected in the EBSD Viewer and read
  back the Confidence Index (CI) for each, with the simulated Kikuchi bands drawn
  over the experimental pattern.
- **Globally refine** a single PC across all calibration patterns.
- Run **pixel-wise PC calibration** — fit a PC field across the scan on a sparse
  grid and interpolate/extrapolate it.
- Edit detector parameters (PC, tilts, pixel size, binning) directly and see the
  effect live.
- Use a **forward-simulated preview** that renders the expected pattern for the
  current geometry and reports its similarity (NCC) to the experimental pattern —
  a visual + numerical way to break the PC/tilt ambiguity that band-fitting alone
  cannot resolve.

It is backed by the `/api/pc/*` routes (and reads the loaded signal/detector from
the EBSD Viewer's state).

## When to use it

Use PC Refinement after loading a scan in the EBSD Viewer and collecting a few
good patterns, and **before** running a full indexing job. The intended workflow,
shown as a 1-2-3 indicator at the top of the page, is:

1. Set / confirm the **detector**.
2. **Load a phase** (CIF).
3. **Add calibration patterns** (done in the EBSD Viewer).

Then index and refine until the CI is high and the simulated bands line up with
the experimental pattern.

## How to use it — step by step

### Set up

1. **Collect patterns first.** In the EBSD Viewer, navigate to several
   well-diffracting pixels and click "Add current pattern to PC Refinement". They
   appear in the **Selected Patterns** list on the left here. Click a pattern (or
   use the slider) to preview it; the PC crosshair and indexed Kikuchi bands are
   drawn over it. **Remove** deletes the selected pattern.
2. **Load the phase(s)** — click **Load Phase(s) from CIF** to open the
   crystal-library picker (the same tick-box picker as the Indexing page). Tick
   every phase that can occur in your calibration patterns; untick one to remove
   it. A phase that is not in the library can be added with the path field at the
   bottom of the picker (paste or type its full path; in the desktop app there
   is also a **Browse…** button). The field next to the picker button loads a
   CIF as the *only* phase. With several phases loaded they are listed with a ×
   each, and every calibration pattern is Hough-indexed against all of them: the
   best-fitting phase wins **per pattern**. The pattern list shows which phase
   each pattern was indexed as, and after **Global PC Refine** the results show
   how the patterns split between the phases. The forward-simulation preview
   uses the master pattern of the phase the selected pattern was indexed as. Up
   to 8 phases can be used together; every extra phase adds a Hough pass to
   each step of the optimiser, so keep the list to the phases that really occur.
3. **Check the detector** — in **Detector Settings**, the PC (`PCx/PCy/PCz`),
   sample tilt, detector tilt, and azimuthal angle are pre-filled from the file
   metadata. Adjust any value; edits apply reactively (debounced) and re-index
   the current pattern. Click **Apply Detector** to commit. The
   **Pixel Size & Binning** sub-group lets you set binning and detector width to
   derive the physical pixel size. Warnings appear if the sample tilt is far from
   70° or the detector tilt is out of the usual range.

### Choose the reflector families (Hough)

Under **Indexing Settings** each loaded phase has a **Reflector families** card.
It lists the plane families Hough indexing compares the detected bands with
(spacing *d*, relative structure factor |F|, multiplicity) and marks the ones
PyEBSDIndex ignores because their pole is a multiple of an earlier family's
(for aluminium {400} after {200}, {222} after {111}). You can untick a family,
ask for the *N* strongest, add one of your own (a typed `hkl`, or `hkil` for a
hexagonal phase, checked against the crystal) or change the rule that builds the
default list (*min d*, *|F| threshold*). A phase you never touch is indexed
exactly as before. The choice belongs to the phase, not to the page: the
Indexing page and every Hough build of that phase use the same one, and the
simulated bands drawn here are the families that are used. Changing it indexes
the selected pattern again with the new list.

**nBands** is the number of bands taken from each pattern; it is sent to the
backend when you change it. The *method* and *search_limit* fields apply to the
optimiser (*search_limit* only to PSO).

### Index and refine

4. **Index Pattern** indexes the currently selected calibration pattern; the
   result shows its CI and overlays the simulated bands. **Index All Patterns**
   indexes the whole calibration set and reports a global CI plus per-pattern CI
   in the list (colour-coded: green ≥ 0.3, yellow ≥ 0.15, orange ≥ 0.05, red
   below).
5. **Global PC Refine** optimises one PC across all calibration patterns using
   the chosen optimiser (Nelder-Mead or PSO) and search limit from **Indexing
   Settings**. It runs as a background task with a progress bar and **Cancel**
   button; on completion it writes the refined `PCx/PCy/PCz` into the results
   box and the detector spinboxes. **Copy PC values** copies them to the
   clipboard, and (for derived datasets) **Apply PC to parent** propagates the
   refined PC back to the parent dataset.

### Forward-simulated preview (geometry check)

6. When a phase is loaded and SHT master patterns are available, a
   **Forward-Simulated** panel appears under the pattern. Pick an **SHT file** and
   a **bandwidth** (Fast 128 / Standard 256 / Sharp 384). It renders the expected
   pattern for the current trial geometry next to the experimental one and shows
   the **NCC** similarity. Sweep the PC and tilt controls and watch the NCC — when
   the simulated and experimental patterns match, the geometry is right. The
   orientation source (hough / supplied / spherical / identity) is shown so you
   know where the trial orientation came from.

### Pixel-wise PC field

7. In **Pixel-wise PC Correction**, choose **Grid Calibration (fit_pc)** or
   **Extrapolate from Points**, set a **grid step** (or click **Auto** for a
   suggestion), and click **Run Calibration**. This refines the PC at each grid
   point across the scan and reports how many points were valid plus the
   per-component spread; **Show PC Map** displays the per-component min/max/mean/
   span. The default **Global (single PC)** mode performs no per-pixel fit.

### Source drift

8. **Source PC Drift** renders Aztec's own per-pixel PC across the whole scan
   (read from the analysis dataset). This shows the *raw* measured PC drift before
   any refinement — large drift means a single global PC throws away signal.
   **Use as seed** copies that per-pixel PC into the calibration store as a
   starting point. **PC Drift Analysis** reports the spread across your
   calibration patterns (needs at least two).

## Inputs & outputs

- **Inputs:**
  - The loaded EBSD signal and its detector metadata (from the EBSD Viewer).
  - Calibration patterns added from the EBSD Viewer.
  - One or more crystal phases from CIFs.
  - Optional SHT master pattern(s) for the forward-sim preview.
  - User-set geometry (PC, tilts, pixel size, binning) and indexing/optimiser
    parameters.
- **Outputs (in-session):**
  - A refined detector PC and geometry, written into the backend detector state
    so later indexing uses it.
  - Per-pattern and global CI values.
  - An optional pixel-wise PC field (grid calibration result).
  - PC values copied to the clipboard, and PC propagation to a parent dataset.

There is no standalone file export from this page — the refined geometry feeds
the rest of the pipeline.

## Tips & notes

- **Judge the result by the forward-sim NCC and the band overlay, not by CI
  alone.** The NCC preview directly compares a rendered pattern to the
  experimental one and breaks the PC/tilt degeneracy that Kikuchi band-fitting
  cannot resolve on its own. Aim for the simulated and experimental patterns to
  look the same and the NCC to be high (≥ 0.3 is colour-coded green).
- **Match the geometry to the Phase Test.** The detector tilt is carried into the
  preview; if the preview looks rotated relative to the single-pixel Phase Test,
  confirm the sample/detector tilt values are the loaded ones.
- **Sample tilt is usually 70°.** A warning appears if it deviates by more than
  0.5°, since most EBSD acquisitions use a 70° tilt.
- **PSO is slower than Nelder-Mead.** A warning is shown when PSO is selected;
  use it only if Nelder-Mead struggles to converge.
- **Per-pixel PC is intrinsically hard.** Pixel-wise calibration fits a sparse
  grid and interpolates; a full independent PC at every pixel is not robust.
  Prefer a refined global PC plus a smooth field, and check the source-drift map
  to see whether per-pixel correction is even needed.
- **Bandwidth trades speed for sharpness** in the forward-sim preview: 128 is
  fast for sweeping, 384 is sharp for a final check.
- **Backend restarts clear the calibration state** (detector, phase, patterns,
  CI). Re-load the file and re-add patterns after a restart.
