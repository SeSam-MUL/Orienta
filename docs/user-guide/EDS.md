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
chemistry-driven **phase suggestion**, and a **phase-map builder** that works in
two separable steps: the map is first cut into **structures** — groups of pixels
that share an element ratio, with no name attached — and you then decide which
phase each structure is. Several structures may carry the same phase, so a map
with a dozen regions can end up with three phases, which is usually what a real
microstructure looks like.

That separation matters because the two steps fail differently. Grouping is a
measurement and can be checked; naming is an interpretation and needs your
judgement, especially where the CIF library holds several near-identical
candidates. Keeping them apart means a naming mistake never destroys the
grouping, and a regrouping never silently renames anything.

The element-to-chemistry maths runs in the backend `eds_utils` module; all map,
probe, quantify, structure, and phase-map operations go through the
`/api/eds/*` routes.

## When to use it

Use EDS after loading a scan with EDS data (in the EBSD Viewer), and typically
**before indexing**, to:

- See where each element sits across the scan and identify chemically distinct
  regions (matrix vs. intermetallics, precipitates, inclusions).
- Read the exact composition (At.% / Wt.% / counts) at a pixel or averaged over a
  region you draw.
- Get a short list of **candidate phases** that match the measured chemistry, so
  you index against plausible phases rather than guessing.
- Build a **chemistry-based phase map** and hand its phase list and pixel masks
  to the Indexing page for chemistry-guided, phase-selective indexing.
- Work on a sample with **no EBSD data at all**. A file holding only Aztec
  element maps opens straight onto this page, and the chemistry phase map is
  then the result rather than a shortlist — which is why every reading in the
  inspector is a measurement you can check, and why every automatic decision can
  be overruled.

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
6. **Zoom into a map** with **Ctrl + mouse wheel** (towards the cursor, up to
   16×); **drag** to pan once zoomed, and **double-click** to reset that map. A
   plain wheel still scrolls the tile list. The **Zoom** switch in the *All Maps*
   header chooses what a zoom affects:
   - **All** — every tile *and* the composite overlay share one view, so you
     compare the same spot across all elements at once.
   - **Single** — each map zooms on its own; only the map under the cursor moves.

   **Reset zoom** returns everything to 1×. Switching the mode keeps your
   per-map views, and going *Single → All* adopts the map you zoomed last, so
   the view never jumps. The current magnification is shown on each zoomed map.
   Crosshair sync, click-to-quantify, Shift-drag regions and the linescan all
   stay pixel-exact while zoomed.
7. **Hover** any map to get a tooltip with all element values, band contrast, and
   (if a phase map exists) the phase at that pixel, in one lookup. Toggle the
   **Lens** (4× magnifier), the **Linescan** tool (drag a line to plot per-layer
   profiles), and **Export PNG** to save the composite. A **swipe compare**
   splitter lets you wipe between two layers.

### Quantify

8. In the right rail's **Pixel Quantification** box, enter a **Row** and **Col**
   (or click a pixel on any map), then **Quantify** to get the Element / Counts /
   Wt.% / At.% table. The footer shows the At.% sum with a ✓/⚠ check that it
   normalises near 100 %. **Copy** puts the table on the clipboard as TSV.
9. In **Region Average**, type a rectangle (Row/Col start–end) or **Shift-drag** a
   box on a map; the page fills the fields and computes the mean ± standard
   deviation per element over the region.

### Build a phase map from the chemistry

The phase map lives on its own tab. The page has two:

- **Element maps** — the composite overlay, the layer stack and every element
  map side by side. This is where you look at the chemistry.
- **Phase map** — the map, the structure tools and the inspector. This is where
  you turn the chemistry into phases.

They are separate because the phase map brought a tool set of its own and one
screen could not hold both without shrinking the element tiles to uselessness.
**Suggest phases** stays on the element tab: it answers a question about the
pixel under your cursor, not about the map.

The EDS signal alone can carry a phase map. It works in two steps, and keeping
them apart is what makes it usable: **the data says which pixels belong
together, you say what they are.**

#### Step 1 — the map is cut into structures

10. In the **Phase Map** controls, click **Re-classify**. The map is grouped into
    **structures**: sets of pixels with the same element ratios. A structure has
    no name yet, and its colour means nothing beyond telling it apart from its
    neighbours.

    Two settings shape the grouping, and both take effect on the *next*
    classification:

    - **Scale** — how far the composition is averaged before grouping (default
      5 px). This is the most important control on the page. Without it the
      grouping is per-pixel noise: on a real 90×120 scan, eight groups came out
      as 3491 disconnected pieces with a **median size of one pixel**. At 5 px
      the same eight groups form 174 pieces. The cost is boundary resolution —
      features thinner than the box get absorbed — which is why it is a slider
      and not a fixed value.
    - **Structures** — how many groups to cut the map into. Leave it on `auto`
      and the count rises while the groups stay chemically distinguishable
      (at least 2 at% apart on some element) and stops when they start
      duplicating each other.

    Expect *more* structures than phases. Over-grouping is the safe error:
    merging two structures is one click, while recovering a structure that was
    never separated is not.

#### Step 2 — you name them

11. Click a region on the map, or a row in the structure list. The **Structure
    inspector** under the map describes it:

    | Reading | What it means |
    | --- | --- |
    | `9 075 px · 16.85% of the map` | how much area this structure covers |
    | `33 connected pieces (5967 · 1193 · …)` | how many separate parts it is in, largest first |
    | **Composition** `at%` | average over every pixel of the structure |
    | **Composition** `±` | how far the pixels differ from that average — a large `±` means the structure is not one thing, but two, or a gradient |
    | **vs background** `1.0×` | this element is no more common here than anywhere else on the map |
    | **vs background** `> 1.3×` | genuinely concentrated here (below `0.8×` it is depleted) |
    | **Touches** `4.1 at% apart` | the largest single-element difference to a neighbouring structure. A small number across a long shared border usually means one region got cut in two — merge it |
    | **Closest phases** `2.7` | mean at% difference between that phase's formula and the measured composition. Smaller is closer |

12. Click a phase under **Closest phases** to put it on the **whole structure**
    in one action. Several structures may get the **same** phase — that is the
    normal case, and it is why one phase name appears on several rows of the
    structure list. The **`N structures → M phases`** box counts each phase once
    so the collapse stays visible while you work.

#### Step 3 — the phase view

13. Switch the map to **Phases**. Now there is **one colour per phase**: two
    neighbouring regions you gave the same phase become one uninterrupted area.
    Many regions, few phases — the phase count is whatever you decided, not
    whatever the classifier guessed.

    Phase colours are shared with the EBSD phase map, so a phase looks the same
    on both pages. Click a legend swatch to recolour it, right-click to reset;
    the choice is saved.

#### Fixing the grouping

No clustering gets this right on EDS taken during an EBSD session: the
interaction volume is far larger than the features, so a small particle reads as
a dilution gradient rather than a plateau and gets cut into concentric rings.
On a real scan one Si particle came out as three structures at Si 24 / 36 /
52 at%. How much rim belongs to the particle is a judgement, so it is offered as
a control rather than decided for you.

- **Merge with…** — fold another structure into the selected one. The most-used
  tool, for exactly the case above.
- **Split into 2 / 3 / 4** — re-group only this structure's own pixels. Local by
  design: raising the global structure count instead would re-cut every other
  structure as well.
- **Boundary −1 px / +1 px** — push this structure's edge out or pull it in.
  Blind to the chemistry; vacated pixels go to the nearest neighbour, never to
  nothing.
- **Snap edges** — let every boundary relax onto the nearest strong chemistry
  edge (a watershed on the composition gradient, seeded from the structures'
  own interiors, so no structure can vanish or swap identity). The slider sets
  how wide a band around each boundary is put up for re-decision.

Every one of these is undoable, one step, and the **Undo** button names what it
would take back.

#### Painting by hand

**Paint by hand** (collapsed while you are grouping) holds the tools that
belong to the phase view: rectangle, polygon, the seeded **wand**, and
map-wide phase replacement. Pixels set by hand are marked as such — they
survive a re-classify, and a later boundary change will not silently revert
them.

14. Use the **send-to-indexing** action to hand the phase map's CIF filenames and
    pixel masks to the Indexing page for chemistry-guided indexing.

### Suggest phases

15. In **Phase Suggestion**, set Row/Col (or click a pixel) and **Suggest**. The
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

- **Maps follow a crop.** If the active dataset is a cut-out from the EBSD Viewer
  (see `EBSDViewer.md`), the element maps, Band Contrast, hover probe, line scan
  and region statistics all describe the cut-out — the coordinates you see and
  the values you get come from the same grid.
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
- **The phase map survives a restart, the composition does not.** The map and
  its structures are written to a sidecar next to the scan, so they come back
  when you reopen the file. **Split** and **Snap edges** need the composition
  the structures were built from, which is not stored — after a restart they
  say so and ask for a re-classify rather than working on different data.
- **Re-classify keeps your hand edits.** Pixels you painted or assigned by hand
  are carried across, matched by phase name rather than by position in the
  candidate list, so a run over a different phase selection cannot silently
  repoint them at an unrelated phase.
- **A structure's composition lists only the elements that grouped it.** C and
  O take no part in the clustering, so they are left out of the readout rather
  than implying they helped decide anything.
- **Enrichment is measured against this map's own background,** not against a
  reference standard — it is the same quantity the classifier gates on, so the
  readout and the classification cannot disagree.
