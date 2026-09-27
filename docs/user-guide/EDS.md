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
two separable steps: the map is first cut into **regions** — groups of pixels
that share an element ratio, with no name attached — and you then decide which
phase each region is. Several regions may carry the same phase, so a map
with a dozen regions can end up with three phases, which is usually what a real
microregion looks like.

That separation matters because the two steps fail differently. Grouping is a
measurement and can be checked; naming is an interpretation and needs your
judgement, especially where the CIF library holds several near-identical
candidates. Keeping them apart means a naming mistake never destroys the
grouping, and a regrouping never silently renames anything.

The element-to-chemistry maths runs in the backend `eds_utils` module; all map,
probe, quantify, region, and phase-map operations go through the
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
   Quick-mode buttons (Phase / IPF-Z / IPF-X / IPF-Y / BC / CI) replace the
   stack with a single layer. Available "add" options include any indexing result layers (Phase, IPF,
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
   profiles), and **Export image…** to open the export dialog for the composite
   (magnification, scale bar, caption, format; a small map opens at 8× or
   more with the bar on). A **swipe compare**
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
- **Phase map** — the map, the region tools and the inspector. This is where
  you turn the chemistry into phases.

They are separate because the phase map brought a tool set of its own and one
screen could not hold both without shrinking the element tiles to uselessness.
**Suggest phases** stays on the element tab: it answers a question about the
pixel under your cursor, not about the map.

The EDS signal alone can carry a phase map. It works in two steps, and keeping
them apart is what makes it usable: **the data says which pixels belong
together, you say what they are.**

#### Step 1 — the map is cut into regions

10. In the **Phase Map** controls, click **Auto-Classify** (the same button reads
    **Re-classify** once a map exists). The map is grouped into
    **regions**: sets of pixels with the same element ratios. A region has
    no name yet, and its colour means nothing beyond telling it apart from its
    neighbours.

    Two settings shape the grouping, and both take effect on the *next*
    classification:

    - **Smoothing** — how far the composition is averaged before grouping
      (default 5 px). This is the most important control on the page. Without
      it the grouping is per-pixel noise: on a real 90×120 scan, eight groups
      came out as 3491 disconnected pieces with a **median size of one
      pixel**. At 5 px the same eight groups form 174 pieces. The cost is
      boundary resolution — features thinner than the box get absorbed — which
      is why it is a slider and not a fixed value. What that pixel count means
      as a physical width is its own subsection below, and it is not the same
      on two scans.
    - **Regions** — how many groups to cut the map into. Leave it on `auto`
      and the count rises while the groups stay chemically distinguishable
      (at least 2 at% apart on some element) and stops when they start
      duplicating each other.

    Expect *more* regions than phases. Over-grouping is the safe error:
    merging two regions is one click, while recovering a region that was
    never separated is not.

#### Step 2 — you name them

11. Click a region on the map, or a row in the region list. The **Region
    inspector** under the map describes it:

    | Reading | What it means |
    | --- | --- |
    | `9 075 px · 16.85% of the map` | how much area this region covers |
    | `33 connected pieces (5967 · 1193 · …)` | how many separate parts it is in, largest first |
    | **Composition** `at%` | average over every pixel of the region |
    | **Composition** `±` | how far the pixels differ from that average — a large `±` means the region is not one thing, but two, or a gradient |
    | **vs background** `1.0×` | this element is no more common here than anywhere else on the map |
    | **vs background** `> 1.3×` | genuinely concentrated here (below `0.77×` — the mirror of 1.3× — it is depleted) |
    | **Touches** `4.1 at% apart` | the largest single-element difference to a neighbouring region. A small number across a long shared border usually means one region got cut in two — merge it |
    | **Closest phases** `2.7` | mean at% difference between that phase's formula and the measured composition. Smaller is closer |

12. Click a phase under **Closest phases** to put it on the **whole region**
    in one action. Several regions may get the **same** phase — that is the
    normal case, and it is why one phase name appears on several rows of the
    region list. The **`N regions → M phases`** box counts each phase once
    so the collapse stays visible while you work.

#### Step 3 — the phase view

13. **Right-click the map** to export it. The menu offers the **region map**
    and the **phase map** whichever one is on screen, so getting the other does
    not mean switching the view and switching back. Both open the usual export
    dialog — crop, resolution, border, scale bar, caption — and both carry the
    scan's pixel size, so a burnt-in scale bar is correct.

14. Switch the map to **Phases**. Now there is **one colour per phase**: two
    neighbouring regions you gave the same phase become one uninterrupted area.
    Many regions, few phases — the phase count is whatever you decided, not
    whatever the classifier guessed.

    Phase colours are shared with the EBSD phase map, so a phase looks the same
    on both pages. Click a legend swatch to recolour it, right-click to reset;
    the choice is saved.

#### The smoothing width is a physical length

The **Smoothing** slider reads in pixels by default, and a pixel is not a
length. That stops being a detail the moment the same setting is used on a
second scan. On this project's own two test files:

| file | step size | what "Smoothing 5" actually averages over |
| --- | --- | --- |
| SampleB | 0.5 µm/px | a **2.5 µm** box |
| 5182-10Mg | 0.75 µm/px | a **3.75 µm** box |

The same number on the same slider, a 50 % difference in the physical area
being averaged, and nothing on screen saying so. Since the box is what decides
whether a feature survives the grouping or is absorbed into its surroundings,
two runs at "Smoothing 5" on those two files are two different analyses
presented as one.

Two things follow. First, the width **can** be asked for as a length, from the
panel. The smoothing control carries a **`px` | `µm`** switch: on `µm` you type
the width you mean, and the classification converts it against this scan's own
step size, rounded to the nearest whole pixel. Under it the panel states what
the last run resolved that to — *"5 px = 2.5 µm box"*, and both edges
(*"5 px = 2.5 × 3.75 µm box"*) where the X and Y steps differ, because the box
is square in pixels and need not be square in microns.

Precedence is explicit and only one of the two is ever sent: a physical width
wins wherever there is a step size to convert it with, and where there is none
the request falls back to the pixel value and **says so** — an orange line in
the panel, from the machine code `scale_source: "pixels_no_step"`, rather than
a physical width quietly becoming a pixel count, which is the exact failure
this exists to prevent. A per-pixel classification smooths nothing, so the
control is hidden there and the run records `not_applicable` rather than a
width it did not use.

A preset stores the width **as a length** (`scale_um`) when you were working in
microns, and restores the unit along with the number — a recipe written in
microns that came back as a pixel count would be this same defect one step
further down the chain. A preset written before that field existed leaves your
current unit alone rather than silently redeclaring it.

Second, and this one you get either way: the width is now recorded **with the
map**, together with the step sizes it was converted against, so it survives a
reload and a backend restart. It comes back out of the export in both forms:

> Recorded with the map: the regions were formed on composition smoothed with a
> 5 px box = a 2.5 µm box.

"Smoothing 5" means nothing to whoever reads your report; "a 2.5 µm averaging
box" is a statement about the sample, and it can be checked against the size of
the features being counted. Where the X and Y steps differ the box is square in
pixels and rectangular in microns, so both edges are quoted rather than one
standing in for both. A map classified per pixel records no width at all —
nothing was smoothed — and that is a recorded fact, not a gap.

#### What decides a match — Min score, the ambiguity flag, and the slider that is gone

One slider sits next to the classify button. Everything else below is a
constant rather than a control — it decides how the result is labelled, and
none of it was described anywhere but in a tooltip. A second slider used to sit
here and has been removed; that is its own paragraph, because a control that
disappears needs an explanation more than one that stays.

**Min score** — default `0.3`. Every candidate phase is scored against the
measured composition on a 0–1 scale, the highest score wins, and if even that
best score stays below **Min score** the pixel or region is left
**unclassified** instead of being pushed into the closest bucket. Raise it and
the map keeps only the confident calls and shows more grey; lower it and
everything gets a name, including the pixels that had no business getting one.
Unclassified is a result, not a failure: a pixel with no usable EDS signal
scores 0 and is *meant* to fall out here.

**Tolerance — the slider is gone.** It used to sit next to Min score, and it
was inert: the request model accepts it, the sidecar stores it, and no code
path in the classification ever read it. Its tooltip described the scorer as it
was before 2026-08-19 — the old rule averaged the At.% deviation over the
phase's elements and rescaled it by this number, which let one large deviation
be diluted by several small ones: a phase requiring 11.6 at% Fe was assigned to
pixels measuring 3 at% Fe, on 20.95 % of a real scan. That scorer was replaced
by a renormalised L1 distance with an outright veto for a missing major
element.

Saying "this does nothing" in grey text under a live-looking slider was honest
and still wrong: a control that cannot change a result has no claim on the
panel, so it was removed rather than annotated. In its place stands one
sentence about the control that **does** decide what stays grey, and — in
per-pixel mode, where the old slider used to live — a line telling anyone who
read an older manual that it is not coming back. The value is still sent, still
recorded with the map, and appears in the export as `tolerance` beside
`tolerance_effective: false`, so a stored map still says what it was made with
and nobody has to wonder whether it mattered.

**The ambiguity flag** — `TIE_TOLERANCE = 0.01` (`cif_phase_library.py:48`). A
region is marked **ambiguous** when the runner-up phase, taken **from a
different degeneracy group**, scored within 0.01 of the winner on that same 0–1
scale. Two members of the *same* group scoring alike is not ambiguity —
chemistry genuinely cannot separate them, and flagging every such pixel would
be noise. The value is pinned by tests between a gap of 0.0030 ("must read as a
tie") and 0.0187 ("must decide"). An earlier version of that line said 0.02 and
claimed to match the same calibration; 0.02 is above 0.0187, so it would have
called an undecidable tie exactly the case that had been calibrated as a
decision.

**The 1.3× enrichment line** — `_ENRICHMENT = 1.3` (`chemistry_score.py:270`).
Before a phase that *requires* an element can be assigned, that element has to
sit at least 1.3× above the map's **own** background level for it. The bar is
measured per dataset rather than assumed, so it needs no k-factor: it asks "is
this element concentrated *here*, relative to the rest of *this* map", which is
the question that separates a particle from the matrix. For scale, real
features measure 2.7× (Fe) and 4.4× (Si) on SampleB, 8.4× (Cu) and 4.0× (Mg)
on 7050.

The value was **chosen from a measured table**, not picked by feel. Calibrated
on SampleB, 2026-08-20; the columns are the share of pixels holding an Fe-phase
with under 25 % of its required Fe, and the share of a ground-truth particle
(Fe > 4 at%) still given an Fe-bearing phase:

| enrichment | false Fe-phase | particle kept |
| --- | --- | --- |
| 1.20 | 1.74 % | 98.8 % |
| 1.25 | 0.76 % | 98.9 % |
| **1.30** | **0.27 %** | **99.0 %** |
| 1.35 | 0.13 % | 99.1 % |
| 1.40 | 0.11 % | 99.3 % |
| 1.45 | 0.06 % | 95.4 % |
| 1.50 | 0.04 % | 88.9 % |
| 1.55 | 0.00 % | 81.6 % |

1.30 sits in the **middle of the plateau** rather than on its best single row.
1.40 scores marginally better and is 0.05 away from a cliff that costs 18
points of particle retention; the constant this one replaced had been placed at
exactly such an edge and had to be corrected. The same 1.3× is what the
inspector's **vs background** column compares against, and `0.77×` — its
mirror — is the depletion line.

**The ratio detection floor** — `_RATIO_FLOOR = 0.005`
(`chemistry_score.py:232`), i.e. **0.5 at%** of the renormalised composition. A
ratio between two elements survives the dilution and sensitivity errors that an
absolute at% does not, but only while both elements are actually measurable:
below the floor the logarithm of the ratio is dominated by background and
counting noise. Two consequences, and they differ:

- Inside the **score**, a pair where either element is under the floor is
  skipped, and an element the composition does not show at all is counted as
  *missing* — it halves the ratio term rather than being quietly ignored.
- Inside a **rule or a region definition**, a ratio clause that cannot be
  measured is **undecidable, and undecidable blocks**. It does not pass. We
  cannot assert what was not measured, and a naive band check on unmeasurable
  legs let 1.7 % of pure noise through on SampleB, where the ratio spanned 0 to
  2.5 × 10¹⁰. For rules the floor is combined with the presence floor
  `_ABSENT = 0.012`, so **1.2 at%** is the binding number there.

#### Carbon and oxygen never reach the scorer, and classify says so

Every phase score in this module drops carbon and oxygen and renormalises over
what is left. They stay in the reported At.% — the composition you read still
sums to 100 including them — but no phase is ever matched on them. That is the
right default for a metallurgical scan, where carbon is mostly surface
contamination, and it is quietly wrong for an oxide or a carbide: **Al₂O₃
scored on its aluminium alone is indistinguishable from Al metal.**

Auto-Classify measures whether that matters *on the scan in front of it* and
returns up to three warnings on its response.

- **`scoring_ignores_c_and_o`** fires when C and O together carry **10 at% or
  more on average**, or when **1 % or more of the pixels** carry **25 at% or
  more** of them. The pixel threshold is the floor of the stoichiometric
  family — Al₂O₃ is 60 at% O, SiO₂ 66, TiC 50, Fe₃C 25 — so a scan clearing it
  is showing a real oxide or carbide feature rather than a dusty surface. An
  alumina scan clears the average threshold six times over. The warning names
  the numbers it measured and says what follows: every phase here was matched
  on its metal content alone, so a named metal may be sitting where the oxide
  is.
- **`indistinguishable_after_excluding_c_and_o`** is stronger, and it is not a
  heuristic. Two candidates whose C/O-stripped compositions are within the tie
  tolerance of each other **cannot differ by more than that tolerance on any
  pixel of any scan** — it follows from the triangle inequality on the distance
  the score is built from. The warning names the pairs. Where one of them is
  reported, the other is equally likely, and which one the map shows was
  decided by the tie-break rather than by the measured chemistry. Al₂O₃ against
  Al is the degenerate case: distance exactly 0.
- **`phase_has_no_scoreable_chemistry`** names candidates made *only* of
  elements the scorer discards. Such a phase has nothing left to compare, so it
  scores neutral against every pixel; where it wins, it won by tie-break.

**These warnings are on the API response and are not shown in the app yet.**
If you are working on oxides or carbides, read them from the `warnings` field
of `POST /api/eds/auto-classify`, or take the general lesson: this
classification cannot tell an oxide from the metal it contains, and no setting
on the page changes that.

#### Rules: deciding which phases may compete

When the automatic pick lands on the wrong candidate, the **Phase rules** tab
under the map lets you constrain it. A rule decides one thing: **whether a
phase may compete for a region** — not how well it scores. A phase whose rule
fails simply cannot win there; the region goes to the next candidate, or stays
unclassified and says which rule blocked it.

That matters because the ambiguity is real but local. On a real scan, two of
seven regions decided between their top two candidates on under 1 at% of
margin while the other five were clear by 1.3 to 3.7 — so rules are per phase
and opt-in, and they leave the calls that were already right alone.

The fastest way to write one is **Rule from this region →**: select a
region on the map, and the rows arrive filled in from what it actually
measures (mean ± twice the spread, on the elements that are genuinely
concentrated there). You then correct the numbers instead of inventing them. A
seeded rule always brackets its own region, so it cannot fail on the region
it came from.

Three kinds of clause, and **every line must hold**:

- **Element range** — `Mg 10–50 at%`. Evaluated on at% renormalised over the
  measured elements with C and O left out, the same composition the inspector
  shows. Never on the at% as it arrives: the element maps sum to 100 *with* C
  and O in them, so once those are dropped the remainder sums to anything from
  91.3 to 100.0 (measured on one scan). An identical composition therefore
  drifts by up to ~8.7 % relative depending on how much carbon a pixel picked
  up, and a band on those numbers would pass in one region and fail in another
  for a reason that is not chemistry.
- **Element ratio** — `Mg:Si 0.5–3.0`. Both elements must clear the detection
  floor; below it the ratio is meaningless (measured: it spans 0 to 2.5 × 10¹⁰,
  and a naive band admits 1.7 % of pure noise) and the rule blocks rather than
  passing.
- **Enrichment** — `Si ≥ 2× background`. Usually the better instrument, and the
  one that answers the interaction-volume problem directly. Measured on a real
  particle: `Si ≥ 2× background` kept **96 %** of it while `Si ≥ 40 at%` kept
  **46 %**. It carries no k-factor and no stoichiometry assumption because the
  bar is the map's own median, which also makes it portable between datasets —
  the factor is stored, the background is re-measured. Element ranges stay
  necessary for the matrix element itself, which is at 0.5× by construction.

**Matrix element.** Above the rule table you can name the element the sample is
made of. Left on automatic it is inferred, which is right most of the time; name
it when the inference would be wrong, because the scoring is built around this
choice and getting it wrong inverts the whole measure. The matrix element is
exempt from the penalty for unexplained elements and from the enrichment bar,
but it stays in the composition — dropping it and renormalising lifted a pure
aluminium matrix from Si 1.2 to 46.3 at% on a real scan, which would have made
the matrix itself look like a phase.

Rules apply in **both** grouping modes, and take effect on the next
classification. A rule naming an element this dataset did not measure blocks
rather than silently passing — you cannot assert a composition you do not have.
If rules exclude every candidate for a region, it stays **unclassified**: there
is no silent fall-back to the automatic pick, because that would make a
rule-driven map indistinguishable from one where the rules never fired.

#### Fixing the grouping

No clustering gets this right on EDS taken during an EBSD session: the
interaction volume is far larger than the features, so a small particle reads as
a dilution gradient rather than a plateau and gets cut into concentric rings.
On a real scan one Si particle came out as three regions at Si 24 / 36 /
52 at%. How much rim belongs to the particle is a judgement, so it is offered as
a control rather than decided for you.

- **Merge with…** — fold another region into the selected one. The most-used
  tool, for exactly the case above.
- **Split into 2 / 3 / 4** — re-group only this region's own pixels. Local by
  design: raising the global region count instead would re-cut every other
  region as well.
- **Boundary −1 px / +1 px** — push this region's edge out or pull it in.
  Blind to the chemistry; vacated pixels go to the nearest neighbour, never to
  nothing.
- **Snap edges** — let every boundary relax onto the nearest strong chemistry
  edge (a watershed on the composition gradient, seeded from the regions'
  own interiors, so no region can vanish or swap identity). The slider sets
  how wide a band around each boundary is put up for re-decision.

Every one of these is undoable, one step, and the **Undo** button names what it
would take back.

#### Defining the regions yourself

Everything above lets the automatic grouping decide where the regions are and
then corrects it afterwards. Sometimes that is the wrong end to start from,
because what separates two regions is **one specific element** rather than the
overall composition. A user hit exactly that case: pure silicon and AlFeMnSi
particles sitting in the same aluminium matrix. The grouping measures distance
over the whole composition, an ~85 at% aluminium background dominates it, and
the few at% of iron that actually tells the two apart is a rounding error next
to it. No cluster count fixes that, because a count cannot say *which* element
carries the distinction.

The **Define regions** tab, next to the region inspector, offers two answers.

**Element weights** keep the grouping automatic and tell it what to care
about. A weight multiplies one element's contribution to the distance; nothing
you see reported changes, because the compositions are still the measured ones.
It does not always help — measured on SampleB, weighting Fe/Mn/Si by 3 moved
the map from 7 regions to 8 and left coherence at 0.933 — so it is worth trying
before reaching for the heavier tool, not instead of it.

**Definitions** take the decision away from the fit. You state the composition
window and the pixels inside it are a region.

**Start with the pixel, not with the region.** Press **From a pixel…**, then
click the middle of the particle the grouping keeps lumping together. The
window is written from what that pixel actually contains, and only over the
elements genuinely concentrated there. This is where the numbers come from: a
threshold for a silicon particle has to be read off a silicon *pixel*, not off
a region that already failed to separate silicon from anything else. Seeding
from the lumped region brackets the mixture and separates nothing — it is the
single most common way to conclude, wrongly, that the feature does not work.

Then press **Try it**: the pixels that window claims light up on the map. That
picture is the point. A count cannot tell you whether you caught the right
pixels — 202 px of matrix and 202 px of particle read identically — so widen
or narrow the number until the highlight matches what you can see. One click
on one particle usually finds *every* particle of that chemistry, which is
what a composition-defined region is for.

The seeded threshold reaches well below the clicked pixel's own value on
purpose. The interaction volume makes a small particle read as a dilution
gradient rather than a plateau, so a window drawn tightly around the core
catches only the core. Measured on a real silicon particle of 202 px:

| accepted down to | 75% | 60% | 50% | 40% | 30% of the clicked value |
|---|---|---|---|---|---|
| claimed | 75 px | 126 px | 157 px | **196 px** | 225 px |

The default lands on the feature; below that it starts taking the
surroundings. How much rim belongs to the particle stays your call, which is
why the number is an ordinary editable field with a live preview rather than
a constant you cannot see.

The controls:

| | |
|---|---|
| **From a pixel…** | arm the map, then click a pixel that is clearly the thing you want. The best starting point. |
| **+ Add** | an empty definition, to fill in by hand |
| **From selected region** | start from a region you picked on the map. Useful when the region is already roughly right; useless when it is the mixture you are trying to split. |
| clause types | **content** is an absolute window in at%; a **ratio** is one element per another; **enrichment** is relative to this map's own background and is usually the sturdiest — on a real particle a 2× enrichment window kept 96% of it where an absolute threshold set to match kept 46%. |
| **↑ ↓** | order is priority, and it is the whole conflict rule — the first definition that matches a pixel keeps it |
| **Phase** | optional. Empty means "group these pixels, then let the matcher name them"; set means you have already decided and the matcher must not overrule you |
| **Try it** | count what each window would claim on this map, without changing anything |

A definition with no clause in it claims **nothing**, and says so. Reading a
half-written window as "the whole map" would wipe the map on the way to typing
the first threshold.

The region inspector has **Define this region by composition →** at the bottom:
you are already looking at the region the grouping got wrong, and one click
turns it into a window you can widen or narrow.

Once definitions exist, the **Regions** count in the tool column renames itself
to **Rest into** — the declared regions are already fixed, so the number applies
to whatever they did not claim. Definitions and weights are stored with the map,
so a reload or a backend restart brings them back rather than leaving an editor
that cannot explain the map next to it.

**Group everything undeclared automatically** is the switch between the two
ways of working. On, the definitions claim their pixels and the rest of the map
is clustered as usual — declare the one or two regions the fit keeps getting
wrong and leave everything else alone. Off, the map is exactly what you
declared plus one region holding the remainder, which is the fully manual case.

Measured on the real SampleB scan, with the two windows
`Si ≥ 20 at% and Fe ≤ 1` and `Fe ≥ 2 at%`:

| | automatic | with definitions |
|---|---|---|
| the Si particle | three regions, Si 25 / 38 / 56 at% | **one** region, 202 px |
| the AlFeMnSi | 6 223 px | 6 857 px |
| named as | Si.cif / Al.cif mix | **Si.cif** and **sd_0302719.cif** |
| coherence | 0.933 | 0.938 |

So the definitions fixed the separation *and* the concentric-ring problem in
one step, which the merge tool would otherwise have had to undo by hand.

#### Painting by hand

**Paint by hand** (collapsed while you are grouping) holds the tools that
belong to the phase view: rectangle, polygon, the seeded **wand**, and
map-wide phase replacement. Pixels set by hand are marked as such — they
survive a re-classify, and a later boundary change will not silently revert
them.

15. Use the **send-to-indexing** action to hand the phase map's CIF filenames and
    pixel masks to the Indexing page for chemistry-guided indexing.

### Suggest phases

16. In **Phase Suggestion**, set Row/Col (or click a pixel) and **Suggest**. The
    chemistry at that pixel is matched against your CIF library, or against a
    built-in fallback library if you have not built one yet. A badge says which
    was used, and the two do not offer the same detail: a **CIF DB** candidate
    shows formula, space group, crystal system and a match score, while a
    **Default** candidate carries only a phase name, a score and the expected
    composition — the fallback library holds no crystallography to show. The
    "on the map" comparison is likewise CIF-only. If you want space groups here,
    build the CIF database.

### Analysis presets

Everything the phase map was built from is a recipe: a smoothing width, a
region count, element weights, and above all the windows and rules you wrote by
hand. Until now that recipe lived in one browser tab beside one scan, and
carrying it to the next sample meant retyping it from memory. The **Preset** bar
at the top of the Phase map controls turns it into a file.

**Apply** · **Save as…** · **Import…** · **Export file…** · **Delete**. One
preset is one JSON file in `%APPDATA%/Kikuchipy/eds_presets/` — readable,
diffable, mailable, and small enough to commit next to a paper draft. Presets
shipped with the app live beside the program and are marked `built-in`: they
cannot be deleted, and saving under one of their names writes **your own copy**
into your preset directory rather than modifying the shipped file.

Saving over a preset you already have bumps its `version` rather than pretending
nothing changed. Saving a preset that constrains nothing is refused outright —
it would mean "whatever the defaults happen to be" on every future dataset,
which is not a recipe.

#### The three that ship

| Preset | What it is for |
| --- | --- |
| **Find the particles in a single-metal matrix** | 2.5 µm smoothing, no pinned matrix. *"GOOD FOR: precipitates and inclusions a few micrometres across in a single-metal matrix — steel, copper, nickel, aluminium. NOT FOR: layered samples or heavily mixed microstructures, where 'matrix plus particles' is the wrong picture to start from."* |
| **Find the particles in an aluminium matrix** | The same, with **Al pinned** as the matrix element — so it refuses a scan whose dominant element is not aluminium *"rather than measure enrichment against the wrong baseline"*. |
| **Large features only (heavy smoothing)** | 5 µm smoothing. *"NOT FOR: fine precipitates. Anything thinner than about 5 µm is averaged into its surroundings and will not appear at all. If you cannot find a particle you know is there, this preset is the reason."* |

All three state their smoothing as a **length**, not a pixel count, so a
built-in means the same physical analysis at any step size: 2.5 µm is a 5 px
box on the 0.5 µm test scan and a different pixel count on yours, and the
analysis is the same one either way. Each carries its own measured numbers in
its notes — on that scan the 2.5 µm setting gave 7 distinct areas at a
coherence of 0.933 and 74 connected pieces, against 50 pieces at 5 µm.

Every one of them says the same thing about itself, and it is worth repeating
here: **this is a sensible starting point you can then correct, not the correct
answer.** And none of them pins a phase list, because none of them can know
your library — so all three raise `phase_list_not_pinned` (below). That makes
them convenient and *not* reproducible. Once you know which phases you care
about, save your own copy with the list pinned.

A built-in cannot be deleted, and saving under its name writes **your own copy**
into your preset directory instead of touching the shipped file. Delete your
copy and the original reappears, which is the point.

#### What travels, and what deliberately does not

**Saved:** the mode, the smoothing width — **as a length (`scale_um`) when you
were working in microns**, which is the one form that means the same analysis
at two step sizes, and as the pixel value otherwise — the region count *and
whether you pinned it* (storing only the number would turn every automatic
preset into a pinned one the moment it was saved — the file records this as the
region count itself: no number means "let it choose", a number means "use
this"), Min score, element weights, region definitions, phase rules, whether
the remainder is clustered, the pinned matrix element, and the explicit list of
phases taking part — never "all", because "all" would silently grow to include
phases the library gained after the preset was written.

Saved beside the recipe, and only as a record of where it came from: the
**element list this scan measured** and its **step size**. They are the entire
input to two of the warnings below, so a preset without them carries guards
that can never fire. Neither is invented — if no file is open, or the file
carries no step size, the field stays empty and the absence is itself reported
(`portability_not_checkable`).

**Not saved, and that is the feature rather than an omission:** file paths, the
grid shape, the crop window, hand-painted pixels, region→phase assignments by
id, and measured background levels. Every one of those is bound to one scan.
Region id 4 exists on the next dataset too and means something else there;
carrying it across is the silent-wrong-answer failure mode, so it is left
behind instead.

**Read that list as a promise about presets the app saved, not as a rule the
file format enforces.** Two keys are actively stripped by the server whatever
sends them — `keep_manual_edits`, and the flag the dialog uses to remember
whether you pinned the region count. Everything else on the list is absent
because the **Save** button never puts it in the payload. A preset written by
hand or posted to the API can carry any extra key it likes: unrecognised keys
are kept verbatim on purpose, so a preset from a newer build survives a round
trip through an older one rather than being quietly thinned. If you hand-edit a
preset, that is the freedom you are holding — nothing downstream will notice a
scan-bound value you put in yourself.

**Not carried either: `Tolerance`.** It is **excluded from the content hash**
and **Apply ignores it**, even in a hand-written file. A preset saved from the
app therefore always shows the default `15`, because nothing in the app writes
this field — but a value you type into the JSON yourself round-trips
losslessly and unchanged, deliberately, against the day the slider is
reconnected. It is inert (see above), and a preset that moved it would have you
tune a control, see nothing change, and conclude the preset was broken. Leaving
it out of the hash matters for the same reason:
a knob that cannot change a result cannot tell two recipes apart, so hashing it
would make two identical analyses hash differently after a hand-edit of the
JSON.

#### Identity is the hash, not the name

A preset carries a name, a version, an author, a creation date, notes, the
element list it was authored against, and a **content hash taken over the
settings alone**. Renaming it, fixing a typo in the notes or adding a tag does
not move the hash; changing a number, a rule or a window does. Two files both
called "6xxx extrusion" are not evidence that two maps were made the same way —
two files with the same hash are. The short hash is shown beside the preset in
the bar and is written into the export, so a map and a recipe can be matched up
afterwards without trusting a filename.

#### The compatibility check is what makes a preset safe

A preset that applies silently to a dataset it does not fit is worse than no
preset at all: it produces a plausible-looking map from a recipe that cannot
mean here what it meant there. So **Apply** checks first and applies second.
Three conditions refuse outright:

| Refusal | Why it is a refusal and not a warning |
| --- | --- |
| The preset names an element this scan did not measure | A clause on an unmeasured element cannot be decided. Undecidable blocks — you cannot assert a composition you do not have |
| The preset pins a matrix element that is not this scan's dominant element | Enrichment is measured *against* the matrix, so getting it wrong inverts the whole measure |
| A phase the preset names is missing from the CIF library | The preset names its phases explicitly, so applying it would run a different phase list than the one it was written for |

A worked example, measured. Take a steel preset that pins **Fe** as the matrix
and carries one rule — *the phase `Cr23C6` needs `Cr ≥ 15 at%`* — and apply it
to the aluminium test scan (measured elements Al, C, Cu, Fe, Mn, O, Si, Zn;
dominant element Al; a library holding `Al.cif`, `Al6Fe.cif` and `Si.cif`).
**All three refusals fire at once**, and this is the report verbatim —

> Cr was not measured in this scan, but the preset uses it in 1 place(s): rule
> for Cr23C6: Cr >= 15 at%. A clause on an unmeasured element cannot be
> decided, so it blocks rather than silently passing.

> The preset pins Fe as the matrix element, but this scan's dominant element is
> Al. Enrichment is measured against the matrix, so getting it wrong inverts
> the whole measure.

> The phase library does not contain: Cr23C6.cif. The preset names them
> explicitly, so applying it would run a different phase list than the one it
> was written for.

The count in the first line is the number of **clauses** the element appears
in, not the number of problems: one window is `1 place(s)`, and a rule that
also weighted Cr, or compared it against another element as a ratio, would say
`3` and list all three. Each blocker names the element and lists **every clause
it appears in**, so you know what to fix rather than only that something is
wrong. Nothing is applied.
The blockers stay on screen as a panel rather than a toast, because a message
that vanishes in half a second is a message nobody read. You can still override
with **Apply anyway and record it** — that is your right, and the refusal then
travels into `provenance.json` with the next export, so the result stays
defensible instead of merely looking clean.

**The report says which file it was checked against, and when.** A stamp
reading `ok: true` and nothing else is authoritative-looking and unfalsifiable:
nothing in it says *which* scan passed. Apply a preset to scan A, switch to
scan B, classify and export, and the record would carry a clean bill of health
for a file that was never checked — a false "compatible" stamp, which is worse
than none. So every applied report now carries `checked_file` and `checked_at`,
and the attestation is dropped outright the moment the loaded file changes.
Two defences for one failure, because the second makes a stale stamp
self-evident if it ever survives by some other route.

#### Warnings, and why an absolute window is the least portable thing you own

Warnings do not stop the apply; they tell you what quietly changed meaning.

- **`absolute_window_not_portable`** — the preset contains one or more absolute
  at% windows, and they are named. Measured on one scan in this project,
  `Si ≥ 2× background` kept **96.3 %** of a real particle while `Si ≥ 40 at%`
  kept **46.3 %**. An enrichment clause stores a *factor* and re-measures its
  bar against each dataset's own background, so it travels between samples; an
  absolute at% window stores a number that was true of one sample's
  composition, and largely does not.
- **`element_set_differs`** — this scan measured a different element set than
  the preset was written against, **including when it measured more**. At.% is
  renormalised over the measured elements, so one extra element shifts every
  other element's at%, and every absolute window therefore means a different
  composition here without a single number in the preset having changed.
- **`step_size_differs`** — the preset was authored at a different step size.
  Smoothing is a physical length (above), so the same setting covers a
  different area here.
- **`matrix_not_checked` / `phases_not_checked`** — the dominant element or the
  CIF library could not be read, so that check was **skipped rather than
  passed**, and it says so.
- **`phase_list_not_pinned`** — the preset never named the phases it may choose
  from, so it runs against whatever the CIF library holds *at the time*. Add
  one CIF next month and the same preset, at the same name, the same numbers
  and the same content hash, quietly runs a different analysis; two runs a
  month apart would only be distinguishable by diffing their provenance
  afterwards. It is a warning and not a refusal, because an unpinned preset is
  perfectly usable — it is just convenient rather than reproducible, and
  refusing it would make the three built-ins unusable on the day they are most
  wanted. Note the trap it names explicitly: a preset **with rules looks
  pinned and is not.** A rule decides which candidates may compete; it does not
  remove the rest of the library from the competition. The same message comes
  back as a hint the moment you save an unpinned preset, rather than waiting
  until somebody applies it — and the app never pins the list on your behalf,
  because "the phases my library happened to hold on the day I pressed Save" is
  a different recipe from the one you wrote.
- **`portability_not_checkable`** — one of the guards could not run at all.
  `element_set_differs` needs the element list the preset was authored against;
  `step_size_differs` needs the authoring step size *and* a step size on this
  scan. Missing any of those, the warning names the guard, what is missing, and
  why it matters, and ends: *"This is not a clean result — it is an unchecked
  one."* A check that silently did not run reads exactly like a check that ran
  and found nothing, which is the whole failure this module exists to refuse.

The Save-as dialog **does** now record what the preset was authored against —
the measured element list and the step size of the scan you saved it from — so
`element_set_differs` and `step_size_differs` fire on a preset saved from inside
the app. Neither field is invented when the scan cannot supply it: no file open,
or no step size, and the field stays empty and `portability_not_checkable` says
which guard is therefore asleep. A preset somebody sent you keeps **their**
authoring record; applying it here never stamps yours onto it, because that
would forge exactly the provenance the field exists to carry.

#### The verdict that reaches the file is the server's, not the browser's

The compatibility report that lands in `provenance.json` is **recomputed at
export time**, against the preset named in the request and the scan actually
loaded — its measured elements, its dominant element, the phase keys this
machine's library holds, its step size. Whatever the client posted as
`compatibility` is **discarded for the record**.

That is not distrust for its own sake. A service-lab tester posted

```json
{"ok": true, "preset_hash": "DEADBEEF",
 "_forged": "this report was computed against a DIFFERENT steel scan"}
```

as the `compatibility` field of the export request, got a 200, and found that
object written verbatim into `provenance.json` — junk key and all — beside a
`source.path` naming another scan. The browser guard was the only defence and
the backend never looked, although it knew the preset name and had every input
the check needs.

**Recomputing beats validating**, and the forgery shows why: she did not only
falsify `ok`, she invented a key no validator would have thought to reject. A
validator has to get every field right **including the ones that do not exist
yet**. A recomputation has to be right once, and the whole class of invented
keys stops mattering, because the posted object stops being an input to the
record at all.

What that means when you read the file:

- **An override records the server's blockers**, not the client's account of
  them. `preset.override` is derived from the recomputed `ok`, so overriding
  documents the objection this server raised. `override_requested` says
  separately whether the caller itself acknowledged a refusal — that is a real
  fact about what you chose, and the only part of the posted object worth
  keeping.
- **A disagreement is kept, not reconciled.** When the two verdicts differ,
  the caller's is recorded beside the server's under
  `compatibility_client_reported`, flagged by `client_report_agrees: false`.
  Quietly reconciling them would hide the one thing worth knowing when there is
  one. An identical copy is not stored — the flag already says they agreed.
- **The client's report is projected onto the fields a report actually has**
  (`ok`, `preset_name`, `preset_hash`, `blockers`, `warnings`; blockers and
  warnings keep `code` and `message`). Anything else is dropped and only its
  **number** survives, as `undeclared_fields_dropped`. "The caller sent one
  field this format does not define" is a fact about the request; its
  content is a fact about nothing, and copying it in would let a caller
  write a sentence of its own choosing into the evidence under a name of its
  own choosing.
- **Where nothing could be recomputed** — no preset named, a name that no
  longer resolves on this machine, a check that could not run — the record
  carries **no `ok` key at all**, with a note saying why. An unverifiable claim
  is not a pass and not a failure. A preset name that has stopped resolving
  never blocks the export either: the map is real, and refusing to write it
  over a bookkeeping problem would destroy work.

The dialog's verdict and the file's verdict come from **one** reader of the
loaded scan, so the panel cannot show you one answer and the folder record
another.

**Where to find the preset's version number.** It is recorded, but not where
you would first look: the path is
**`preset.compatibility.preset_version`**, *not* `preset.version` — the
top-level `preset` block carries `name`, `compatibility` and `override` and no
version of its own. Grep for `preset.version`, get nothing, and it is easy to
conclude the version was never written; it was. It is the number the file
itself carries, so it tracks the same way — first save 1, every re-save one
higher — and it is the number to quote in an email. It sits inside the
compatibility record because that record is what the server recomputes at
export time against the preset it actually resolved, so the version stated is
the version that was really read. One consequence follows from that: the field
is written **only when the preset was resolved and checked**. Name no preset
and there is no compatibility record at all; name one that no longer resolves
on this machine and you get the record with its note explaining why nothing
could be checked, but no version — because nothing was read to take a version
from. The content hash beside it is the stronger identifier of the two; the
version is the one a human quotes.

#### The file on disk

One preset is one JSON file, UTF-8, keys sorted, **LF line endings**, and the
bytes `Export file…` hands you are byte-identical to the file this build saved
for the same preset. They were not: saving went through Python's text mode and
landed CRLF on Windows while the export returned LF, so a tester who diffed a
mailed copy against her saved one measured **81 of 81 lines differing with not
one character of content changed** — exactly the diff that makes somebody
believe the preset travelled wrong. The shipped built-ins are LF too, so one
convention covers all three routes, and the export response states the
separator rather than leaving you to guess at a diff.

**A preset saved by an older build is still CRLF, and stays that way until you
save it again.** Only new writes go through the fixed path; nothing rewrites
your existing files behind your back, because silently rewriting a user's files
on read is worse than leaving them alone. The consequence is narrow but it is
exactly the workflow the fix was for: mail an *old* preset with `Export file…`
and diff it against your own copy and you will still see every line differ,
because the mailed bytes are LF and the file on disk is not. The remedy is one
action you already have — open it and **Save** it once. Measured on a CRLF file
written by the old path: 36 CRLF line endings before, none after, the content
hash unchanged, and the file then byte-identical to what `Export file…` hands
you. The only thing that moves is the **version number**, which increments as
it does on any save. Until you do it the file is not broken in any other way:
it loads, applies, exports and hashes identically either way.

**Tags are yours to write, and they are what finds a preset later.** `Save as…`
takes a comma-separated tag line and a material class alongside the author,
notes and date; the preset picker then searches over name, material, tags,
author and notes, and offers the material classes and tags present as filters.
Both fields are free text on purpose — a service lab expects sixty to a hundred
presets after a year and will group them by whatever it actually works on
("6xxx extrusion", "customer 4711", "Symmetry S2"), and any vocabulary we could
write down would be wrong within a week. Tags are de-duplicated
case-insensitively, because the filter compares in lower case and two chips
selecting the same presets is not a choice. Adding or editing a tag does not
move the content hash: it does not change the analysis.

### Getting the numbers out

**Export data…**, the full-width button just below **Auto-Classify**, writes
the whole analysis into a folder: not a picture of the map, but the counts, the
areas, the compositions, and the record of how they were produced. It is
disabled until a map exists, because there would be nothing to count. The
folder is named `<scan>_EDS_<YYYY-MM-DD_HHMM>` and is **never** written into
one that already exists — an export is evidence, and overwriting one silently
replaces a record somebody may already have cited.

The dialog counts before it writes, so it can tell you what you would get
before you get it.

#### A note on the example numbers

Everything below is illustrated with measurements from one test scan,
**SampleB** — 90×120 pixels, 0.5 µm step. They are not all from the *same
export*, and they are not meant to add up against each other. A phase map is
the product of its settings: change the smoothing width, the region count or
the region definitions and the regions change, so the particles change, so
every count downstream changes with them. Three states appear in this manual,
and each number is labelled with the one it came from:

| State | What it produced |
| --- | --- |
| **six regions**, exported whole | 133 particles, 36 of them touching the edge |
| **seven regions carrying five phases**, nothing painted | the plausibility figures and the per-region margins |
| **cluster mode, 1.5 µm smoothing**, nothing painted | 135 particles — the dissent and id-renumbering figures |

If you re-run any of these on your own copy of SampleB with your own settings
you will get different integers, and that is the point of the note rather than
a defect in it. **Read every example as the shape of an answer, never as a
property of the sample.**

#### What each file answers

| File | The question it answers |
| --- | --- |
| `phases.csv` | **How much of each phase**, plus a real `unclassified` row |
| `regions.csv` | **The evidence behind that** — every region, what it measured, which phase it went to, and by what margin |
| `particles.csv` | **One row per connected particle** — sizes, shapes, positions, compositions |
| `definitions.csv` | Every **region definition**, verbatim and in one readable line, and how many pixels it actually claimed on this map |
| `provenance.json` | The full run record. Always written |
| `caption.txt` | The one-liner you paste under a figure. Written whenever the run record has a summary line to write — which is every ordinary export, but unlike `provenance.json` it is not unconditional |
| `phase_map.png`, `region_map.png` | **The picture the app draws**, one pixel per scan point, so a headline number can be eyeballed against the map that produced it |
| `phase_map_figure.png`, `region_map_figure.png` | The same pixels magnified for a report (a small map 8× or more, nearest neighbour, so every colour is the 1:1 file's) with a scale bar when the scan step is known; `provenance.json → map_figures` records the factor and the bar length. Omitted only when it would be the 1:1 file again |
| `summary.xlsx` | All of the above, one sheet per table plus a flattened provenance sheet — for reading; the CSVs are the machine-readable copy |
| `labels.npz` | Opt-in. `region_id` / `phase_id` / `particle_id` rasters plus the lookup arrays that give the ids meaning, so somebody else's script can reproduce a figure pixel for pixel |
| `pixels.csv` | Opt-in, large. One row per pixel — the artefact that lets every other table be re-derived and checked |

`labels.npz` and `pixels.csv` are off by default because neither is needed to
answer "what phases, how much, how big", and `pixels.csv` grows with the scan:
a half-million-pixel map with a dozen elements produces a file no spreadsheet
will open.

Two notes on `definitions.csv`. Your **phase rules** are not in it — they are
recorded verbatim in `provenance.json` instead, under the classification
settings. And each definition's claim is **re-measured** at export time rather
than read back from a stored count, because the map may have been merged,
split, grown or edge-snapped since; that re-measurement runs on the *raw*
composition while the classification ran on the *smoothed* one, which is why
the `definition_match_frac` on a region row is an attribution strength and sits
a little below 1 near boundaries rather than being a bookkeeping identity.

`provenance.json` is not optional. It carries the app version, the source file
with its size and SHA-256, the crop window if the scan is a cut-out, the step
sizes, the measured element list, the elements excluded **from scoring**
(carbon and oxygen — they stay in the at% renormalisation; see *Two answers on
every row*), the background level each `enrichment_*` column was divided by,
the requested **and** resolved value of everything automatic ("asked for auto,
resolved 8"), the classification settings **as recorded with the map** rather
than as echoed by the dialog, every rule and definition verbatim, the ambiguity
threshold actually used, both rankings with their units and populations spelled
out, the determinism statement, and the semi-quantitative caveat as literal
text. An export without it is a screenshot.

It now also carries, each with its own subsection below: `acquisition` (what
the scan was taken with) and `interaction_volume` (how big the volume behind
each measurement is), `library` and `phases_absent` (which candidate list ran
and who was excluded from it), `plausibility` (the file's check on its own
result), `coordinates` (the pixel convention, spelled out where a script will
read it), `peak_overlaps`, `hand_edits` including the edit log, and `summary` —
the caption, also written as `caption.txt`.

#### `FORMAT_VERSION`, and what moves it

`provenance.format_version` is **3**. It exists so a downstream sheet can pin
it, and it follows one rule: **a column appended at the end of a table does not
move it; a column that changes meaning or disappears does.** That is the whole
value of the number — a sheet pinned to an older version keeps working when a
column is added, and breaks loudly when one is redefined under a name it
already knew.

Version 3 carries **one redefinition**: `particles.assignment_source` is graded
on the particle's own `n_px_hand` instead of inheriting its region's grade
(above). Everything else in it is an addition and would not have moved the
number on its own — `chemistry_dissents`, `third_phase`/`third_score`,
`phase_label`/`formula_ascii`, the per-phase particle-size summaries, the
plausibility columns, and the `acquisition`, `interaction_volume`, `library`,
`phases_absent`, `coordinates`, `peak_overlaps`, `summary` and
`hand_edits.log` blocks in the provenance.

Version 2 was the round before: `phase_auto`/`phase_agrees` moved to the scorer
that made the map, `perimeter_um` became `perimeter_crofton_um`, a particle's
`margin_at_pct` became `region_margin_score` beside the particle's own, and
`assignment_source` gained `mixed` and `untracked`.

#### `region`, `phase` and `particle` are three different questions

- A **region** is a chemistry group spanning the whole map. It has no name of
  its own.
- A **phase** is a name, and several regions can carry the same one — which is
  why a phase row and a region row rarely have the same count.
- A **particle** is one connected component: one physical object you could
  point at in the image.

None of the three substitutes for another. "This phase covers 12 % of the map"
(phase), "it did so as 9 chemistry groups" (regions) and "it did so as 214
separate particles" (particles) are three different statements about the same
pixels, and a report that mixes them up is wrong in a way a reader cannot spot.

Particles are connected components **of a region**. On a map classified per
pixel there are no regions, so the phase grid stands in and the export warns
you: two *touching* regions carrying the same phase name are two particles
under the region grid and one under the phase grid, so the two counts are not
comparable. Whichever was used is stated in the
provenance as `particle_basis`, and the neighbourhood that defines "connected"
(**4-** or **8-connected**, 8 by default) is recorded beside it, because it
changes the count.

Measured on the SampleB test scan, an export of the whole map produced
**133 particles across 6 regions**, of which **36 touch the edge**. That
particle count was checked against an independent `scipy.ndimage.label` run on
the region raster in `labels.npz` and matched exactly. Both numbers move with
the classification settings, so they are an illustration of the shape of the
answer and not a property of the sample.

#### The three denominators, and which one you want

Every fraction carries its denominator in its own column name. There is no
`area_pct`, because half the readers of an `area_pct` guess the wrong
denominator and the three differ by tens of percent:

| Column | Denominator | Adds up to |
| --- | --- | --- |
| `frac_of_scan` | every pixel of the exported raster | **1** across all phase rows *including* `unclassified` |
| `frac_of_valid_eds` | the pixels carrying a usable EDS measurement | **1** across all phase rows *including* `unclassified` |
| `frac_of_classified` | the pixels that received a phase | **1** across the named phases; empty on the `unclassified` row |

**Quote `frac_of_scan` unless you have a reason not to**, and quote the
`unclassified` row beside it. It is the only one whose rows add up to the image
you are showing, so a reader can check it against the picture.
`frac_of_classified` is a phase's share of the *named* area — the larger number
whenever anything is unclassified, and honest only if you are willing to assume
the unclassified area is distributed like the classified area, which is exactly
the assumption that fails when a phase is missing from your library.
`frac_of_valid_eds` is the one to reach for when part of the map carries no
usable signal at all.

**Each fraction's numerator comes from the same set as its denominator**, and
`frac_of_valid_eds` is where that has teeth. Its numerator is the row's pixels
that *themselves* carry a usable EDS measurement — the column `n_px_valid_eds`,
which sits immediately beside it — and not the row's total `n_px`. A row can
own pixels with no EDS at all (a hand-painted one, for instance), and counting
those inflated the fraction and stopped the column closing. It **does** total 1
now, to floating-point rounding, across all phase rows including
`unclassified`, because the phase masks partition the raster and every valid
pixel therefore lands in exactly one row. That is worth knowing as a check: a
`frac_of_valid_eds` column that does not sum to 1 is a bug, not a property of
your data.

**A count sits next to every fraction** — `n_px`, or `n_px_valid_eds` for
`frac_of_valid_eds` — so a binomial interval is recoverable and you are never
asked to trust a percentage without its count. Nothing is rounded at export:
rounding is presentation and belongs in whatever reads the file.

The same intersection rule governs the summary scalars in
`provenance.counts`: `n_px_classified_and_valid_eds` is counted the way every
row is counted, so the scalar and the tables cannot disagree.

#### Two answers on every row, and which one decides

Every region row and every particle row carries the name it ended up with
(`phase_final`) **and** the name the chemistry alone would give it
(`phase_auto`), with `phase_agrees` comparing the two in one column. Never
overwrite the software's answer: hiding a disagreement is what gets a report
discredited.

That only works if `phase_auto` is the answer that actually made the map, and
until format version 2 it was not. The map is named by the scorer in
`chemistry_score.score_phase_ratio` — with your phase rules applied as a gate,
the matrix element exempt, a Min-score floor, and each region measured on its
**eroded interior**. That replay runs on **raw** composition, because the
naming step never used the smoothed values: smoothing feeds the feature matrix
that decides which pixels group together, and the group is then named on the
composition as measured. The export recomputed something else entirely: an
unweighted mean absolute deviation over the union of element keys, with none of
that. Two metrics, and the second one's answer was printed in a column called
`phase_auto` beside a boolean called `phase_agrees`.

The consequence was not academic. On the real 90×120 SampleB scan — 7 regions,
22 candidates, **not one hand-painted pixel** — four of seven regions read
`phase_agrees = False`. A reader can only understand that as "a human moved
this one", and a tester duly drafted a Methods sentence claiming a manual
override that never happened. It is now **0 of 7**: the export replays the real
scorer, on the same population, and reproduces `region_phase` for 7 of 7
regions of that scan.

So the columns now mean this:

| Column | What it is |
| --- | --- |
| `phase_auto`, `score_auto` | the deciding scorer, replayed. 0–1, higher is better |
| `runner_up_phase`, `runner_up_score` | the next candidate under that same scorer |
| `margin_score` | winner minus runner-up. How far clear the call was |
| `third_phase`, `third_score` | **third place**, because a margin cannot be read without it |
| `phase_agrees` | does the map match the replay |
| `phase_second_opinion_mad`, `gap_mad_at_pct`, `runner_up_mad`, `margin_mad_at_pct` | the old ranking, under names that say what it is |

**Third place is in the file for one reason.** A margin of 0.15 between first
and second means one thing when third is far behind and another when third is
0.01 under second — the second case is a three-way tie, and without the third
column the file reported it as a decision. It is on the deciding metric only; a
third-place *second opinion* would be noise.

The mean-absolute-deviation ranking survives deliberately, because a distance
in at% is easier to judge than a 0–1 score and because it is what the region
inspector shows on screen. It decides nothing. `phase_auto_index` is `-1`
wherever `score_auto` falls below Min score — exactly what the classifier does
with such a region — while `score_auto` is still filled in, because *"the best
chemistry could manage was 0.21 and the floor is 0.30"* is the sentence you
need and blanking it would hide it. And `phase_agrees` is a plain equality
including the −1 = −1 case: a region the map left unclassified and the
chemistry would also leave unclassified is an **agreement**.

**Every object is ranked on its own composition.** A particle used to inherit
its region's margin: on the real scan a 286 px and a 175 px particle printed
the identical `0.366300`, which was their 660-pixel region's — and a number set
beside a 175-pixel object is read as describing it. Each particle now carries
its own ranking, and the region's sits beside it under the name
`region_margin_score`, which says whose it is. A particle whose chemistry
dissents from the region it sits in is exactly the row worth finding, and
`chemistry_dissents` is the column that finds it — see *Who decided this row*.

**Both sides of the second opinion are now on one basis.** The at% maps are
renormalised over *every* measured element, carbon and oxygen included — the
per-pixel sum is 100.000000 everywhere, measured. Drop C and O, as every scorer
does, and the remainder runs 91.30 to 100.00. A CIF's nominal composition
contains no C or O and sums to 100 by construction, so subtracting one from the
other charged every phase for a few at% of carbon it could not contain, and
charged it unevenly across the map. The measured mean is now renormalised onto
the scored basis before the subtraction. On the same scan that changed no
winner, but moved a margin by up to **+112 %** (region 4: 0.1947 → 0.4140 at%)
and changed one **runner-up's identity** (region 6). A margin is the number
that says whether a call was a decision or a coin toss, so a factor of two in it
is not cosmetic.

An earlier note in this manual said C and O were excluded from the
renormalisation. They are not. They are excluded from **scoring**, and the two
statements are different — a reader who believed the old one read every at%
cell against the wrong denominator. `scored_at_pct_sum` in each table is the
at% that survives the drop, so how much of the signal the scorer never saw is
visible per object.

One consequence to expect rather than to be surprised by: the file's second
opinion and the region inspector on screen can differ in the third digit, and
rarely in the runner-up. The panel shows the metric **without** the
renormalisation, rounded to 2 dp with elements below 0.05 at% dropped. The
file is the corrected one.

#### The columns whose names do not fully define them

| Column | Why it is there |
| --- | --- |
| `phase_label` | a name a figure legend can carry. `phase_name` is `sd_0302719.cif` and `formula` is `Mn₄.₅₁₂Al₁₂₇.₂₉₆…` — a database id and a refined site occupancy printed as a formula, and neither can go in a legend. The CIF's filename stem is preferred, with database bookkeeping (`_mp-570001`, `_symmetrized`) removed; when the stem is only an id it carries no chemistry, so the label is built from the composition — elements in descending at%, hyphen-joined — with the id kept in brackets so the row stays traceable. Never invented: with neither to work from the cell is empty and you fall back to `phase_name` |
| `formula_ascii` | the same formula with its Unicode subscripts flattened to digits. A subscript survives a CSV and does not survive a plotting label, a LaTeX run or a shell |
| `particle_area_mean_um2`, `particle_area_max_um2`, `particle_area_sd_um2` | the size distribution behind `n_particles`, on the phase row, without grouping `particles.csv` by hand. Mean **and** max **and** sd, because on this kind of map the mean is dominated by a cloud of near-resolution objects and the max is the one a failure analyst reads. The sd is empty for a single particle rather than `0` — one object has no spread, and `0` would read as "they are all the same size" |
| `sqrt_area_um` | Murakami's inclusion-rating parameter — the length a failure analyst rates an inclusion by. **Not** interchangeable with `ecd_um`: for a disc the two differ by a factor 0.886, and further for anything elongated |
| `feret_angle_deg` | the orientation of `feret_max_um`: degrees from the **+x axis** (increasing column), **counter-clockwise as seen on the map**, folded into [0, 180) because a diameter has no head or tail. The row index increases downward and the row component is negated accordingly — without that every angle in the file is mirrored, and *"are the inclusions aligned with the extrusion direction"* gets the wrong answer with full confidence |
| `enrichment_<El>` **and** `background_at_pct_<El>` | how enriched the object is over this map's own background, **with the divisor in the next column**. This project has shipped an "Al 49.68× enriched" readout on a map whose background *is* aluminium, and seeing the divisor is the only defence a reader has against a repeat |
| `scored_at_pct_sum` | the at% that survives the C/O drop. A region sitting at the bottom of the 91–100 range is a region a tenth of whose signal the scorer never saw |
| `perimeter_crofton_um` | renamed from `perimeter_um`, so the correction survives somebody opening the CSV without the JSON. Same number, honest name |
| `n_px_valid_eds` | the numerator of `frac_of_valid_eds`, beside it |
| `region_margin_score` | the region's margin on a particle row, under a name that says whose it is |

The angle carries one caveat worth reading before you plot a rose diagram:
`feret_max` on a rectangle is its **diagonal**, so a 1×7 bar reports 171.87°
rather than 180°. The angle is the direction of the longest caliper, which is
what `feret_max_um` is the length of — measured from the same convex hull,
scaled the same way, so the length and the direction describe one line. For an
elongated object the two converge; for a stubby one they do not, and
`aspect_ratio` is the column that says which case a row is. The cell is empty
where the hull was unavailable, because an angle guessed from a bounding box
would be a different quantity.

**What is deliberately absent: `raw_at_pct_sum`.** A pre-renormalisation total
— raw counts, or wt% before conversion — is not recoverable at the point where
these tables are built and is therefore missing rather than silently
approximated. The export is handed at% maps only; the counts → wt% → at%
conversion happens upstream and renormalises to 100 on the way, so no function
in the exporter has ever seen the un-normalised total. `scored_at_pct_sum` is
the recoverable neighbour, and the provenance says exactly this in place of a
column that would have to be invented.

#### The file checks itself, and changes nothing

A group leader used a finished export to sanity-check the science and found the
headline wrong: a large share of an aluminium extrusion classified as
Fe-intermetallic, by phases whose assigned pixels carry **a third of the iron
those phases are made of**. Both numbers were already in `phases.csv`, in
adjacent columns — `mean_at_pct_Fe` and `nominal_at_pct_Fe` — and the file said
nothing about the difference. Her sentence: *"You already print both numbers
next to each other. You are one subtraction away from saying so."*

The check is that subtraction. **It is a report on the result and it changes
nothing** — no assignment, no score, no ranking, no grid; `phase_final` is
identical with it and without it. The classification has exactly one scorer,
and a second one wearing the word "check" would be the two-metrics-in-one-file
defect this export was repaired for.

**The criterion is relative, and it has to be.** The elements that decide a
phase identity are the small ones: on the SampleB scan Fe runs 0.16 to
4.27 at% while Al runs 80 to 97. An absolute at% threshold meaningful for Fe
would fire on every row for Al, and one quiet enough for Al could never fire
for Fe at all. So:

1. The row's measured composition is put on the **scored basis** (C and O
   dropped, renormalised) so it shares a denominator with the CIF's nominal,
   which is itself renormalised over the elements this scan measured.
2. Only **discriminating** elements are compared — the ones the phase needs at
   **≥ 2× the map's own background** and at **≥ 1 at%**. Below that factor the
   element cannot separate the phase from the matrix: a pixel of pure
   background already reads more than half of what the phase requires.
3. The worst such element is reported as a **factor**, symmetric in direction
   (`deficient` or `excess`), with a floor of 0.05 at% on the denominator so a
   measured zero gives a large finite ratio rather than an infinity.
4. `composition_implausible` is `True` at **≥ 2×**. It is `None`, never
   `False`, where there was nothing to check.

The 2× threshold sits in a measured gap. On SampleB the best-fitting checkable
element on any assigned phase reaches 1.15×, and the smallest ratio a diluted
or wrong call produces is 2.06× — and 2.0 is placed at the *top* of that gap
deliberately, because under a factor of two on a semi-quantitative at% there is
nothing to say.

**What it found on the real scan.** All the figures in this section come from
one map and only that one: the 90×120 SampleB scan classified into **seven
regions carrying five phases, with nothing hand-painted**. Other numbers
elsewhere in this manual were measured on other settings of the same scan and
will not add up against these — see *[A note on the example numbers](#a-note-on-the-example-numbers)*.
The two rows the group leader challenged:

| Phase | Element | Measured | Nominal | Ratio |
| --- | --- | --- | --- | --- |
| `Al6Fe_mp-570001_symmetrized.cif` | Fe | 4.03 at% | 14.29 at% | **3.55× deficient** |
| `sd_0302719.cif` | Fe | 4.35 at% | 11.60 at% | **2.67× deficient** |

Both clear the 5 %-of-scan floor (27.64 % and 33.45 %) and get their own named
warning. Every implausible row, big or small, is flagged in its table and
counted in the summary warning: that export assigned **five phase rows, of
which four were checkable, and all four came out implausible, together
covering 62.56 % of the scan** (33.45 + 27.64 + 1.24 + 0.23). The floor exists
so that the other two — a 134 px silicon particle and a 25 px sliver — stay
row-level curiosities instead of moving a headline. The silicon one is
instructive and is reported rather than tuned
away: those pixels really are silicon, and their measured Si is halved because
the interaction volume is larger than the particle. "Deserves a look" is the
true statement about it.

That every checkable row on this scan failed is itself the finding, and it is
worth reading the right way round. It does not mean the phase names are wrong;
it means **not one of them can be corroborated on the element that identifies
it**, on a scan whose particles are mostly smaller than the volume each
measurement comes from. The check points; it never certifies. And the largest
row on the map — the aluminium matrix, over a third of the scan — is the one
it cannot check at all.

**And `Al.cif` reads `plausibility_checked = False`.** Not "fine" — *not
checkable*: no element of pure aluminium stands far enough above an aluminium
matrix's background to testify about the assignment, and `plausibility_note`
says so in the cell. An absence of evidence is written as an absence of
evidence.

Warnings are raised from `phases.csv` only. A region and the phase it carries
are largely the same pixels, so warning from both tables would state one
finding twice and double the fraction of the map it claims to describe.
`regions.csv` carries the same per-row columns, which is where you go for the
detail.

**`review_flag` and `review_reasons`** OR together everything the row can
answer, as codes rather than prose so a boolean with three causes stays
sortable: `implausible_composition`, `undecided_between_phases` (the winning
score within 0.02 of the runner-up — chemistry did not decide this row), and
`untracked_assignment`. Only reasons the row *has columns for* are considered:
the phases table has no margin and no assignment source, so it contributes the
composition reason alone. A missing column must not read as "fine".

#### Who decided this row

`assignment_source` is graded rather than asserted, because "a human touched
this" is a claim and the file should only make claims it can back:

| Value | What it means |
| --- | --- |
| `hand` | every pixel of the row was painted by hand |
| `mixed` | *some* were. `n_px_hand` beside it is the count. This case used to read `hand`, so 100 painted pixels claimed a 3327-pixel region |
| `definition` | a composition window claims more than half of the row's pixels; `definition_name` and `definition_match_frac` say which and how strongly |
| `auto` | nothing was painted **and** the map matches the replayed chemistry. Only then can a row positively assert that nothing touched it |
| `untracked` | nothing was painted and the map does *not* match — or the replay produced no answer at all. Something moved it that leaves no trace in the stored map |

`untracked` is the honest value for a region whose phase you gave it by hand:
naming a region deliberately does not lock anything, so a renamed region is
indistinguishable in the stored map from a classified one — except that the
chemistry no longer agrees with it. That is the one signal available, and it
produces `untracked` rather than a positive assertion of `auto`. A hand-painted
pixel outranks a definition, because painting is the most explicit statement in
the file about where a phase boundary sits. (Where the edit log exists, it
resolves the ambiguity `untracked` cannot: an entry with `op: name_region`
tells "somebody renamed it" apart from "the replay produced no answer".)

**On a particle row, `hand` and `mixed` are graded on that particle's own
`n_px_hand`.** They used to be copied wholesale from the region, and a QA
tester measured the result: **110 of 167 particle rows read `mixed` while
carrying `n_px_hand = 0`**, with exactly one particle in the whole file
actually holding a painted pixel. The file's own legend defines `mixed` as
"some pixels were painted", so 110 rows of an auditor's table asserted a human
decision that did not happen. A particle with none of its own painted pixels
can now never read `mixed`.

`definition`, `auto` and `untracked` *are* still inherited from the region, and
that is not the same mistake: under region basis the phase name genuinely was
decided per region and the particle was never named individually. They are
recomputed with the region's own painting zeroed, so a human who touched a
different particle in the same region cannot leak back into this row. This
redefinition is why `FORMAT_VERSION` moved to 3.

**`chemistry_dissents` is a second column because it answers a second
question.** `assignment_source` is a *provenance* claim — who moved this row.
`chemistry_dissents = True` says only that **this particle's own composition
would name a different phase from the one it carries**; nobody touched it. It
is the same fact as `phase_agrees = False`, spelled so it cannot be misread as
"a human changed this" — and the two facts genuinely part company at particle
level, which is exactly why one cell could not carry both.

Expect dissent often, and on a clean map: a region is named from its **eroded
interior mean**, and each connected component of that region is then measured
**separately**, so a component sitting on the region's chemical edge dissents
as a matter of course. Measured on the SampleB scan in cluster mode at 1.5 µm
smoothing, with nothing painted — the 135-particle state, not the 133-particle
one used above — **94 of 135 particles dissented, and every one of them was
provenance-clean.**
On a *region* row a disagreement does carry provenance weight — that is the
`untracked` case above. On a particle row it does not. Read
`chemistry_dissents` as "look at this object's chemistry", never as "look at
who edited this object".

#### What was done to this map by hand

Seven ways exist to put a thumb on a phase map — paint pixels, name a region,
merge, split, grow, snap edges, and replace one phase across the whole map —
and exactly one of them, painting, used to leave a trace. The map now **counts
all of them**: `merges`, `splits`, `grows`, `edge_snaps`, `regions_named`,
`paints`, `phase_replacements`, tallied as each edit happens and carried in the
sidecar, so they survive a reload and a backend restart. `paints` is the one
counter that covers more than one gesture: the rectangle, the polygon and the
magic wand all land in it, because all three write pixels the same way. Which
it was is kept in the edit log entry, not in the counter. The individual entries are kept too, oldest first, **capped at 500
with the true total beside them** — a truncated log that looks complete is the
exact failure this record exists to prevent, so when the two numbers differ the
record says so. A re-classify starts a fresh log and fresh counters; it is not
itself a hand edit, and recording it as one would put a false entry in
somebody's provenance.

**A map made before this existed reports `null`, never `0`.** "Nobody was
watching" and "nothing happened" are different answers, and collapsing them
would rebuild the hole the feature fills. Once a map *is* tracked every counter
is a real integer including the zeros — "0 merges" is then a measurement, which
is what lets a reviewer answer the question a group leader put plainly: *"if the
merges field is blank I cannot tell whether they merged nothing or merged forty
times until the answer looked right."*

**One warning that follows from this, and it will bite a figure.** Merging
renumbers region ids — the dropped id is removed and everything above it shifts
down by one — and particle ids are ordered by region id, so **particle ids move
too**. A figure labelled with particle ids and re-exported after a merge, a
split, a grow or an edge-snap no longer points at the same particles. The map
records that its region grid has been edited for exactly this reason. Re-export
and re-label together, or label after the last boundary edit.

**All of it is now in the export.** `provenance.hand_edits` carries
`pixels_painted` as it always did, and beside it `merges`, `splits`, `grows`,
`edge_snaps`, `regions_named`, `paints`, `phase_replacements`, their sum under
the older name `boundary_edits`, the true total `n_edits` — and the **edit log
itself**, one entry per edit with the operation's own detail, oldest first.
An earlier version of this manual said the counts were tracked and thrown away
on the way out. They were, for one release; they are not now.

Two notes on reading it. `log_retained` says how many entries are listed and
`n_edits` how many there were; `log_truncated` says whether the two differ. And
`paints` counts paint
*operations since the last classification* while `pixels_painted` counts the
pixels currently carrying a hand assignment, whenever they were painted — so
`paints: 0` beside `pixels_painted: 25` is the ordinary reading of a map that
was painted and then re-classified, not a contradiction. Both notes are in the
file beside the numbers.

Where a boundary edit has happened, the export also raises the warning
`region_ids_renumbered` rather than leaving it to a boolean nobody reads: a
tester measured, on that same 135-particle map, 58 of 135 particle ids
changing after a single merge at
1.000 pixel overlap — the ids moved although not one pixel did.
`hand_edits.region_grid_edited` is the field, the warning is what makes it
visible, and `null` there still means "this map carries no edit record", never
"no such edit happened".

One limit remains, stated rather than left to be discovered: the counts are
served by the phase-map API but are **not shown anywhere in the app**. Read
them from the export, or from `GET /api/eds/phase-map`.

#### Reading the particle table without fooling yourself

Four columns are load-bearing rather than decorative.

**`touches_edge`.** A particle cut off by the raster border is truncated, and
its size is a **lower bound**, not a size. On the SampleB export that is 36 of
133 particles. Exclude them from any size distribution, or state that you did
not — a histogram that silently includes them is biased low, and the bias grows
with the particles.

**All-pixel vs `core_` composition — the most useful warning in this
document.** The EDS interaction volume during an EBSD session is *larger than
the small particles you are measuring*, so the rim pixels of a small particle
are particle-plus-matrix mixtures rather than particle. Averaging a particle's
composition over all of its pixels therefore drags every small particle toward
the matrix and **manufactures a "composition depends on particle size" trend
that is not in the sample** — one of the easiest wrong results to publish,
because it looks like physics. Both are exported: the plain `mean_at_pct_*`
columns over all pixels, and `core_mean_at_pct_*` measured after a one-pixel
erosion, with `n_px_core` beside it so you can see how much particle is left.
Use the core composition for small particles, and say which one you used.

**`sd_within_at_pct_*` is spatial spread, not a measurement uncertainty.** It
is the population standard deviation of the composition across the pixels of
that object, so it mixes genuine heterogeneity — a gradient, or two things
grouped as one — with counting noise, and the two are not separated. It is
named `sd_within` precisely so it cannot be read as an error bar. **No standard
error is exported, deliberately:** the pixels are strongly spatially correlated
(the interaction volume exceeds the step, and the classification box-averages
on top of that), so `sd/√n` would overstate the precision by one to two orders
of magnitude. You get `sd_within` and `n_px`, and you decide.

**`below_size_limit` flags, it never deletes.** *Flag particles smaller than
N px* marks the small rows and leaves them in the file, counted. Silent
exclusion would quietly halve a particle count with nothing in the export
saying so, so the count you export is always the count that was measured.

**Both exclusion counts are numbers, not prose.**
`provenance.counts.n_particles_below_size_limit` and
`n_particles_touching_edge` sit in the record as integers beside
`n_particles`. They used to exist only inside an English warning string, which
meant a caption builder had to parse a sentence — fragile, and untranslatable.
Neither removes a row; both are flags. Together they are what you need to state
the exclusions honestly: *"133 found; 0 below the limit; 36 touching the edge;
97 used."* Both integers come out of `provenance.counts`, beside
`n_particles`, so a caption builder reads three numbers rather than parsing an
English sentence. The export dialog previews the particle, region and phase
counts before you write anything; the exclusion breakdown is in the folder, not
in the dialog.

One more in the same spirit: `perimeter_crofton_um` is a 4-direction Crofton
estimate rather than a boundary-pixel count, because counting staircase pixels
overestimates the perimeter by 4/π = 1.273 at every size, which reads as a 38 %
*under*estimate of circularity. The name carries the correction so it survives
somebody opening the CSV without the JSON. It is still meaningless for a blob
of a few dozen pixels — check `n_px` before quoting a shape number.

**A fifth thing, and it is not a column.** The matrix has particles too — and
they are not particles. On the six-region export used above, `Al.cif` carries
**16 of them**: connected components of the *continuous* matrix, left over
between the second-phase objects. The rows are not wrong, they are counted
exactly as every other phase's are, and what is wrong is calling them particles
— which the column heading cannot say and the file therefore does. How large
that number gets is entirely a property of how fragmented the matrix is on the
map in front of you: the tester whose complaint produced this warning was
reading a **different** export of the same scan and had `phases.csv` telling
her "143 Al particles", which she said a student would paste straight into a
thesis. The export now raises `matrix_counted_as_particles`, naming the phase
and whatever the count happens to be. The matrix phase is identified by the **matrix element the
classification itself resolved** and a nominal composition of that element
alone — `Al.cif`, not `Al6Fe` — rather than by area, because a genuinely
dominant intermetallic in a small field of view would trip an area rule.
**Filter on `phase_name` before quoting a particle count or a size
distribution.**

#### What the scan was taken with, and how big the volume behind a pixel is

The caveat above — *the interaction volume is larger than the small particles
you are measuring* — was prose, and a reader could not act on it. It is now two
numbers.

**`provenance.acquisition`** is read straight out of the scan's own
`/…/EDS/Header`: beam voltage, working distance, magnification, process time,
energy range, channel width and count, tilt, detector elevation and azimuth,
detector serial and type, window type, drift correction, frame count, and the
acquisition date and labels. The named fields carry the vendor's unit in their
name (`beam_voltage_kv`, `working_distance_mm`, `channel_width_ev`); the whole
group also rides verbatim under `acquisition.raw`, so a key this exporter does
not know about is preserved rather than dropped. Where no header can be read,
`available` is `false` and a warning says so — nothing is defaulted, because a
beam voltage nobody measured would be worse than none.

This is what the semi-quantitative caveat has to be weighed against. The file
asserts "no k-factor calibration"; you cannot judge that claim without knowing
the kV, and until now the kV was in the scan and never in the export.

**`provenance.interaction_volume`** turns the kV into a length: the
Kanaya–Okayama electron range in the **matrix element the classification
resolved**, at the recorded voltage —

> R = 0.0276 · A · E^1.67 / (Z^0.89 · ρ)   µm

On the SampleB scan that is **4.187 µm at 20 kV in aluminium, against a 0.5 µm
step** — a factor of eight — and **120 of the 133 exported particles, 90 %,
have an equivalent circle diameter smaller than the range.** That single line
is the one to read the composition columns next to.

The comparison is on `ecd_um` because the range is the diameter of a roughly
spherical volume, so diameter against diameter is like for like;
`feret_max_um` would flatter an elongated particle and `sqrt_area_um` is a
different rating parameter. And the range is a **bound on the volume, not a
resolution** — X-ray generation happens inside it, so the effective sampling
volume is smaller. A particle below the range is measured together with its
surroundings: its composition is pulled toward the matrix, and the pull scales
with size, which is precisely how a spurious composition-versus-size trend gets
manufactured. The `core_*` columns mitigate that. They do not remove it.

`range_um` is `null` when either the beam voltage or the matrix element is
unavailable, and for an element with no density on hand — an invented density
would put a made-up length in a report.

#### Micrometres, or nothing

Areas are `n_px × step_x × step_y`, exactly — no interpolation, no fitted
scale.

**When the scan carries no step size, every µm column is omitted entirely:**
`area_um2`, `ecd_um`, `feret_max_um`, `perimeter_crofton_um`, the lot. Not zero, not a
guess, not a 1 µm/px default — the columns are simply not there, and the reason
is written into the provenance. A size distribution computed on a guessed scale
is worse than a missing one, because it looks like an answer. Pixel counts and
compositions are unaffected, and the dialog tells you which case you are in
before you export.

The same principle governs elements: a cell for an element this scan never
mapped reads **`not measured`**, never `0`. "Cu 0.0 at%" would assert that
copper was looked for and found absent, which is a different and false
statement. That the cell is text rather than a number is intentional — a
spreadsheet that refuses to average the column is telling the truth about it.
Elements the scan did not measure still get their column, because a reader
comparing this export against a copper-bearing phase has to be able to see that
copper was never mapped, and an absent column looks like an oversight while an
absent value is a statement.

#### Where a pixel is

Worth stating, because it is the kind of thing that is silently assumed and
then wrong by half a step in somebody's overlay. All of it is also written into
`provenance.coordinates`, because `pixels.csv` and the label rasters are the
artefacts somebody else's script reads, and a convention that lives only in a
manual is a convention the script will get wrong.

- **Indexing is 0-based and row-major.** `row` runs 0…`n_rows−1` downward,
  `col` runs 0…`n_cols−1` to the right, and the pixel table walks the raster
  row by row. `bbox_row_min`/`bbox_col_min` and the label rasters in
  `labels.npz` use the same indices, so a CSV row and a blob in the raster are
  the same object.
- **`x_um = col × step_x` and `y_um = row × step_y`, exactly.** No offset is
  added anywhere. The first pixel therefore sits at (0, 0), and every
  coordinate is the offset of *that pixel's* sampling point from the first
  pixel's — which is the convention the particle centroids
  (`centroid_x_um`/`centroid_y_um`) are measured in as well, so the two are
  directly comparable.
- **Y increases downward**, as the raster does — not upward, as a plot axis
  would.

The consequence to watch: if you are registering this export against something
that puts its origin at the *outer corner* of the raster and places pixel
centres at `(col + ½) × step`, you have to add half a step. The export does not
do it for you, and half a step is exactly the kind of error that survives a
visual check.

#### Which library ran, and who was never allowed to compete

An auto-classification does not score every phase in your CIF library. A phase
containing an element this scan never mapped is removed **before scoring**,
because its missing elements would be matched against zeros. That removal used
to be silent: a tester counted 30 entries in the library, 22 competing, and
**8 gone with nothing in the file to say so or why**.

`provenance.library` is the identity of the list that ran — path, `n_entries`,
`n_participating`, `n_absent`, file size, mtime, and **two hashes**.

**Why two.** `sha256` is of the library **file**, so it moves on any edit,
including one this export never reads. `entries_digest` hashes only what the
export can see of each entry — key, filename, formula, space group,
composition. A **chemistry or symmetry edit moves both**; an edit to anything
else, a lattice parameter refined between two heat treatments for instance,
**moves only `sha256`**. That is the point of the pair: two exports carrying
the same two hashes were scored against the same candidates, and where they
differ the pair tells you which kind of edit happened. A library that could not
be read at all reports so explicitly — "we could not check" and "nothing was
dropped" are different statements, and the absent list is omitted rather than
written empty.

`provenance.phases_absent` names every entry that did not compete, with a
reason code, because the four are four different actions for you:

| Reason | What it means, and the fix |
| --- | --- |
| `element_not_measured` | the phase contains an element this scan never mapped; `missing_elements` names them. Map the element |
| `not_requested` | the phase is in the library and every element of it was measured, but the run was narrowed to a subset. Widen the selection |
| `not_in_library` | the run asked for a key the library does not have. Add the CIF and rebuild the database |
| `excluded_upstream` | every element measured, not narrowed out, and still absent. Reported honestly rather than assigned one of the three known reasons |

Measured on the SampleB scan: **30 entries, 22 participating, 8 absent — seven
of them for magnesium and one for nickel**, all `element_not_measured`. Where
any phase is dropped for an unmapped element the export also raises the warning
`phases_dropped_unmeasured_element`, so the count reaches you without opening
the JSON.

#### Peak overlaps are declared, not detected

Nothing here measures an overlap — this export receives per-element at% maps
and never the spectra, so it cannot. What it can do is say which known overlaps
are **live on the file in front of you**, and `provenance.peak_overlaps` does
that: a pair is listed only when **both** of its elements were mapped here.

The at% values come from Aztec **window integrals** — counts in a fixed energy
window per element, not a fitted, deconvolved spectrum — and a window collects
every line that falls inside it, whoever emitted it. An over-reported element
is over-reported *everywhere*, so it shifts a phase's measured composition in
one direction rather than scattering it, and it can move a ranking whose margin
is under a couple of at%. Read `margin_score` and `third_score` before trusting
a call that turns on an element listed here.

On this project's Al–Fe–Mn–Si scans two pairs apply, and the first is the one
that matters: **Mn Kβ at 6.49 keV sits under Fe Kα at 6.40 keV, so a fixed Fe
window over-reports Fe in an Mn-bearing alloy** — and Fe is the element that
decides between an Fe-intermetallic and the matrix. The second is the Al Kβ
tail at 1.55 keV running into the Si Kα window at 1.74 keV, which raises the
apparent Si in an aluminium matrix. On a scan with no manganese the first
sentence does not appear at all, because the block is about the file it sits
in. An empty `applies_to_this_scan` means no declared pair had both its
elements mapped here — **never** that the spectra were checked and found clean.

#### The picture, and the caption

Two artefacts exist because a number in a table is not checkable by eye.

**`phase_map.png` and `region_map.png`** are written into the folder by the
**same renderers the page draws with**, decoded from the base64 the phase-map
API already returns — including your phase-colour overrides. They are not a
second drawing: a picture that coloured a phase differently from the screen
would be worse than no picture, because a reader comparing the two would
conclude one of them is a different map. The reader who found the wrong
headline number above said a picture would have caught it in four seconds, so
the images are written by default and there is no checkbox for them. Beside
each 1:1 file sits `…_figure.png`: the same pixels magnified for a report (a
21 × 22 px crop becomes 352 × 336 px) with a scale bar when the scan step is
known — nearest-neighbour, so it can never show a colour the 1:1 file does
not. The export dialog on the Phase map tab is still there for a *figure* you
arrange yourself, with crop, resolution, caption and annotations.
`region_map.png` appears only when the map has regions. A failure to render
costs you a picture and never a number: it warns, writes what it managed, and
every table is unaffected.

**`caption.txt`** is one line you can paste under a figure, assembled purely
from values already in `provenance` — scan, grid and step, mode and smoothing
box, regions → phases, particles and how many touch the edge, percent
classified, the largest phase, kV, the interaction range and the fraction of
particles under it, hand-painted pixels, the library ratio with its hash, the
semi-quantitative sentence, and the app version and export time. It is also in
`provenance.summary`; it is a file as well because a JSON key is not something
anybody pastes under a figure four weeks later. **A clause whose value is
unavailable is left out rather than printed with a placeholder** — every clause
is guarded on "is this value absent", not on "is it falsy", because a step of
0.0 µm, a count of 0 regions and a missing value are three different things and
the on-screen version of this line once shipped a confident `0` for a missing
one. That same guarding is why the file is not promised unconditionally: if
every clause were unavailable the line would be empty, and an empty
`caption.txt` is worse than no file — so none is written. `provenance.json`
always is.

#### A field this format does not define is refused

`POST /api/eds/export` rejects a request carrying a key it does not declare,
with **422**, and the same holds for saving and importing a preset. This is not
strictness for its own sake. A PhD tester sent `scale_px: 3` to an EDS
endpoint, got a **200**, and ran 60 scans smoothed at the default width he
meant to change. Silence is the failure mode: a rejected request costs him a
minute, an accepted one is a wrong analysis he never sees. A field this model
does not declare is a typo, not an extension.

#### Decimal separator and column separator

Both are yours to choose: decimal **point** or **comma**, and **comma**,
**semicolon** or **tab** as the column separator. A German Excel silently
reading `3.14` as `314` is a wrong number in a report, and nothing warns you.

**Comma with comma is refused — twice.** `3,14` written into a comma-separated
file is two fields, and every column after it shifts by one — a file that opens
perfectly and puts a different number in every column. The dialog will not send
the request, and the server refuses it as well with **400** and the message
*"decimal ',' cannot be combined with delimiter ','…"*, so a script or a `curl`
posting straight to `POST /api/eds/export` is stopped by the same rule rather
than getting the broken file. Use a semicolon or a tab with a decimal comma.
`summary.xlsx` is
unaffected either way: it stores real numeric cells, and Excel applies the
reader's own locale to those.

#### What this export does not do

Stated plainly, because the alternative is that you assume otherwise.

- **No batch over a folder.** One scan at a time. The export core is written as
  a pure function of the data — no session, no request — precisely so a batch
  driver can call it later without touching it, but that driver does not exist
  yet and belongs with the Batch page rather than here.
- **No EDS/EBSD agreement check.** The most-wanted missing number, and it is
  missing on purpose: the EDS chemistry prior can *drive* the very indexing the
  comparison would be checked against, so today the number would be circular.
  It needs an independence flag the indexing export does not yet record.
- **No sensitivity re-runs.** Nothing here tells you how the map would change
  at a different smoothing width or Min score. Re-run and compare by hand.
- **No standard errors.** See `sd_within` above — a considered refusal, not a
  gap.
- **No k-factor calibration against a standard.** The composition is the same
  semi-quantitative Cliff–Lorimer-style normalisation used everywhere else on
  this page, and that caveat is written into `provenance.json` as literal text
  so it travels with the numbers instead of staying in this manual.
- **No peak-overlap *detection*.** The overlaps that apply to this scan are now
  declared in `provenance.peak_overlaps` (above), but nothing measures one:
  this export receives per-element at% maps and never the spectra, so an empty
  `applies_to_this_scan` means no declared pair had both its elements mapped
  here — never that the spectra were checked and found clean.
- **No per-element raw counts, and no pre-normalisation total.** The counts →
  wt% → at% conversion happens upstream and renormalises to 100 on the way, so
  no function in the exporter has ever seen the un-normalised total. It is
  declared absent rather than approximated; `scored_at_pct_sum` is the
  recoverable neighbour. If you need raw counts, read them out of the H5OINA.
- **No stable particle ids across a boundary edit.** Ids are dense and are
  ordered by `(region_id, −n_px, centroid)`, so a merge, a split, a grow or an
  edge-snap renumbers them. The export says so — `hand_edits.region_grid_edited`
  and the `region_ids_renumbered` warning — but it cannot give you the old id
  back. Re-export and re-label together, or label after the last boundary edit.
- **No `Tolerance` control, and no tolerance in the result.** The slider was
  inert and has been removed from the panel. The value is still recorded, and
  it is recorded as `tolerance_effective: false` rather than being presented as
  though it had done something.

## Inputs & outputs

- **Inputs:**
  - An open H5OINA scan containing per-element EDS count maps (`/1/EDS/Data/Window
    Integral/<element>`).
  - Optionally your curated CIF library
    (`Database/crystal_database.xlsx`) for phase suggestion and auto-classify;
    optionally an indexing result for Phase/IPF/CI overlay layers.
  - Optionally an **analysis preset** — a JSON file from
    `%APPDATA%/Kikuchipy/eds_presets/`, or one somebody sent you and you
    imported.
- **In-session outputs:**
  - Quantification tables (per-pixel and per-region), copyable as TSV.
  - A stored **phase map** (held per-process) that other tools and the
    send-to-indexing hand-off can consume.
  - A combined and per-phase **pixel mask** + CIF filename list forwarded to
    Indexing.
- **File outputs:**
  - The composite overlay, the region map and the phase map exported as
    **PNG** images.
  - The **data export folder** `<scan>_EDS_<date_time>/` — `phases.csv`,
    `regions.csv`, `particles.csv`, `definitions.csv`, `provenance.json`,
    `caption.txt`, `phase_map.png` and `phase_map_figure.png` (and the
    `region_map` pair where the map has regions), `summary.xlsx`, and
    optionally `labels.npz` and `pixels.csv`. See
    *Getting the numbers out*.
  - **Analysis presets** as single JSON files, saved into
    `%APPDATA%/Kikuchipy/eds_presets/` or written out anywhere with
    **Export file…**.
  - Quantification tables are copied to the clipboard rather than written to
    disk.

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
  its regions are written to a sidecar next to the scan, so they come back
  when you reopen the file. **Split** and **Snap edges** need the composition
  the regions were built from, which is not stored — after a restart they
  say so and ask for a re-classify rather than working on different data.
- **Re-classify keeps painted pixels and discards hand-given region names.**
  This asymmetry is deliberate and it is the one thing on this page that
  surprises people. Painting a rectangle, a polygon or a wand selection **locks**
  those pixels (`phase_map_store.assign_mask`), so they are carried across a
  re-classify — matched by phase *name* rather than by position in the candidate
  list, so a run over a different phase selection cannot silently repoint them at
  an unrelated phase. **Naming a region does not lock anything**
  (`assign_region_phase`): the region *is* the record of that decision, which is
  what lets a later boundary move keep following it — and it is also why
  rebuilding the regions from scratch throws the names away. If you want a
  decision to survive a re-classify, paint it; if you want it to follow the
  region's edge, name it.
- **A region's composition lists only the elements that grouped it.** C and
  O take no part in the clustering, so they are left out of the readout rather
  than implying they helped decide anything.
- **Enrichment is measured against this map's own background,** not against a
  reference standard — it is the same quantity the classifier gates on, so the
  readout and the classification cannot disagree.
- **The classification is deterministic.** Every clustering step is seeded:
  `random_state=0` at all three KMeans call sites and at the Gaussian-mixture
  fit in `eds_clustering.py`, with `n_init=10`. The same settings
  on the same file give you the same map, every time — so a difference between
  two runs is a difference in the settings, not in the algorithm, and a map is
  reproducible from its recorded settings alone.
- **The `±` in the region inspector is spatial spread, not measurement
  uncertainty.** It is the standard deviation of the composition *across the
  pixels of that region*, so it mixes genuine heterogeneity — a gradient, or two
  things grouped as one — with counting noise, and the two are not separated.
  Do not read it as a precision or an error bar on the mean. A large `±` is a
  signal that the region is not one thing; it is not a statement about how well
  the composition was measured.
