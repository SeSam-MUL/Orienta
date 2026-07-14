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

### Fix pseudo-symmetry / wrong orientation (grain flip)

8. For spherical results a **Try variants ⬡** button loads candidate orientations
   for the pixel — the current one, crystallographic pseudo-symmetry variants, a
   Hough candidate, and the **adjacent same-phase grains'** mean orientations
   (badge "↖ neighbour grain") — each rendered with its render-NCC and sorted
   best-first. The neighbour candidates cover *foreign orientation basins*:
   grains whose correct orientation is neither a pseudo-variant of their own
   stored orientation nor the Hough solution — typically the correct
   orientation sits right next door.
9. If the correctly-indexed grain does **not** touch the wrong one, use the
   **Reference pixel** row/col inputs + **Add candidate**: any indexed pixel's
   stored orientation joins the gallery as a rendered, scored candidate
   (badge "⌖ reference").
10. Click the candidate whose simulated pattern matches the experimental one,
    set a **grain threshold** (degrees) and click **Apply to grain**. The
    correction is flood-filled across the connected, same-phase,
    similar-orientation pixels (the grain); each pixel keeps its own measured
    deviation (per-pixel snap / rigid transfer — no single Euler triple is
    stamped onto the grain), and the stored CrystalMap is updated. One-level
    **Undo** is available.
    **Apply to all similar grains** does the same AND fixes every other
    same-phase grain map-wide that sits at the same wrong orientation —
    each sibling is adopted only if its own render-NCC clearly improves
    (genuinely different small grains stay untouched); one Undo restores
    everything. Use it for scattered mis-indexed nests that are painful to
    click one by one.

### Navigate precisely (nudge + neighbourhood zoom)

Tiny nests (2–5 px) are hard to hit by clicking the map. Below the NCC
mini-map both dialogs show a **neighbourhood zoom** (±7 px IPF-Z crop with a
crosshair on the selected pixel) and **arrow buttons** — arrow keys work too —
to step the selection pixel by pixel.

### Assign a different phase (Compare-phases mode, Phase Maps dialog)

When **Compare phases** shows that another phase's simulated pattern clearly
beats the stored one (e.g. an Al line present in the experiment that the
stored phase misses), click **Assign "<phase>" to this grain**: the connected
grain is reassigned to that phase with per-pixel Hough orientations of the
new phase, with one-level **Undo**. The caption above the button shows what
is currently stored. (This is the surgical sibling of the map-wide Phase
Verification tool.)

### Export a figure

11. In the **Phase Test** dialog (Indexing → *Phase Test*), the figure composer
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
