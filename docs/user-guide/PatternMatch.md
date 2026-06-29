# Pattern Match

## What it does

Pattern Match lets you **verify an indexing result pixel by pixel** by placing the
measured (experimental) Kikuchi pattern next to the **simulated** pattern for the
orientation/phase the indexer assigned, plus a per-pixel **NCC difference image**
and a **match score (R)**. It answers the central question after indexing: *did
this pixel actually get the right phase and orientation?*

It has two parts:

1. **The Pattern Matches inspector** — a dialog with three panels (experimental |
   best-match simulated | NCC image), a clickable NCC heatmap to pick the pixel,
   and a large colour-coded R value. For multi-phase spherical results it can also
   re-test the clicked pixel against every phase and step through them.
2. **The publication-figure composer** (`PatternExportDialog`) — a layout editor
   that arranges the experimental / simulated / NCC / heatmap panels with linked
   markers and a scalebar into one figure, then exports it as PNG/JPEG or copies it
   to the clipboard.

For **Spherical Indexing** results the simulated pattern is rendered on the fly
from the `.sht` master via the GPU forward operator; for **Dictionary** results it
comes from the in-memory dictionary. It uses the backend
`/api/indexing/pattern-match*` routes.

## When to use it

Use Pattern Match **after indexing**, to:

- Confirm that high- and low-confidence pixels really match (a high R means the
  simulated bands line up with the measured ones).
- Diagnose **mis-indexing** — for chemically degenerate phases, compare what each
  candidate phase would have looked like at the same pixel.
- Fix **pseudo-symmetry** errors by flipping a grain to the correct orientation
  variant (spherical results).
- Produce a **publication-ready comparison figure**.

## How to use it — step by step

### Open the inspector

1. Open it from the **Indexing** page via **View Pattern Matches** (enabled for
   Dictionary results), or from the **Phase Map** page via **View Pattern Matches**
   — including by clicking a pixel or an entry in the Anomaly Browser, which
   pre-selects that pixel.

### Inspect a pixel

2. The dialog loads the **NCC heatmap** of the whole map on the left. Click any
   pixel (a crosshair marks it). The right side fills with the three panels and the
   R score.
3. The **R value** is colour-coded: green (good, R ≥ 0.30), orange (acceptable,
   R ≥ 0.15), red (poor). The phase name and Euler angles are shown beneath.
4. Use the **rank** stepper (− / +) to step through the next-best matches kept for
   that pixel (Dictionary results keep several).

### Quality and aperture controls

5. For **Spherical** results choose the **Quality** (SHT bandwidth) — *Fast* (128,
   ~225 MB VRAM), *Standard* (256, ~1.7 GB), or *High* (384, ~5.7 GB) — for a
   sharper simulated pattern.
6. Set the **Aperture**: *Auto* (detects black-cornered detectors), *Circular*
   (force a round mask — needed for EDAX phosphor patterns whose corners are
   dark-grey, not black), or *Full* (no mask). With *Circular* a radius slider
   trims the masked area so signal-free corners don't dilute R.

### Compare phases (spherical, multi-phase)

7. Tick **Compare phases** to re-index the clicked pixel against *every* phase's
   `.sht` and step through the per-phase simulated patterns sorted by R (best
   first). This reveals when the indexer's "winner" was a near-tie with another
   chemically similar phase. The first click warms a per-phase cache (slower);
   later clicks are fast.

### Fix pseudo-symmetry (grain flip)

8. For spherical results a **Try variants ⬡** button loads candidate orientations
   for the pixel — the current one, crystallographic pseudo-symmetry variants, and
   a Hough candidate — each rendered with its render-NCC and sorted best-first.
9. Click the variant whose simulated pattern matches the experimental one, set a
   **grain threshold** (degrees) and click **Apply to grain**. The chosen rigid
   correction is flood-filled across the connected, same-phase, similar-orientation
   pixels (the grain) and the stored CrystalMap is updated.

### Export a figure

10. In the **Phase Test** dialog (Indexing → *Phase Test*), the figure composer
    opens with the experimental / simulated / NCC panels for the selected
    candidate. Drag, resize, align and distribute panels, toggle linked markers and
    a scalebar, save/load layout presets, then **export** as PNG/JPEG (with a
    resolution scale) or **copy to clipboard**.

## Inputs & outputs

- **Inputs:**
  - An **active indexing result** (CrystalMap) in backend memory.
  - The corresponding phase files (`.sht` for spherical rendering; the in-memory
    dictionary for Dictionary results) and the detector geometry stored with the
    result.
- **Outputs:**
  - On-screen comparison (experimental / simulated / NCC) and the R / NCC scores —
    informational, no file written by the inspector itself.
  - **Grain-flip:** an in-place modification of the stored CrystalMap (corrected
    orientations for the affected grain).
  - **Figure composer:** an exported PNG/JPEG image (download) or a clipboard copy.

## Tips & notes

- **Judge by R, not just CI.** The confidence index from indexing is a within-phase
  peak measure; the forward render-NCC / R here directly compares simulated vs.
  measured bands and is the more honest correctness check.
- **EDAX patterns:** if the simulated and experimental panels look misaligned at the
  corners, force **Circular** aperture — auto-detection misses the dark-grey EDAX
  phosphor corners.
- **VRAM:** higher SHT quality and Compare-phases both cost GPU memory. If rendering
  fails, drop the quality or use **Release GPU** on the Indexing page.
- **Simulated pattern unavailable:** Dictionary results only render the simulated
  panel when the dictionary is still in memory; after a backend restart re-run
  indexing. Spherical results render directly from the `.sht`, so they survive as
  long as the file path is valid; any render error is shown verbatim in the panel.
- **The grain-flip is universal** — it works for any pseudo-symmetry (trigonal,
  HCP, tetragonal, cubic approximants) because you pick the correct variant by eye
  from the render-NCC-ranked thumbnails, not from a fixed rule.
