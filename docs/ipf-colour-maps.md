# IPF Colour Maps in Orienta — the Complete Picture

**Audience:** developers + scientific users. This document records everything
Orienta does to make IPF (inverse pole figure) orientation maps correct and
readable for low-symmetry phases, why each piece exists, what it changes
(data vs. display), where the code lives, and the evidence behind each claim.

---

## 1. Why this document exists

On a real multi-phase aluminium dataset (Al matrix + Al7FeCu2 + an
alpha-AlFeMnSi cubic approximant, point group m-3) the IPF-Z map looked like
colour noise: one physical grain rendered as a salt-and-pepper mix of distant
hues. Fixing this took **five independent mechanisms**, because the visible
artefact had several *unrelated* causes — some in the **data** (stored
orientations really were wrong) and some purely in the **display**
(orientations correct, colours misleading).

The single most important lesson, encoded in everything below:

> **When a map "looks wrong", first separate data from display.** Export the
> orientations (`.ang`) and measure neighbour disorientations offline *before*
> touching any algorithm. On the reference dataset this proved the
> orientations were smooth (all neighbour pairs < 5° under m-3) while the map
> still looked broken — pointing squarely at the colour key, not the indexing.

---

## 2. The pipeline at a glance

```
 spherical GPU indexing (SHT SO(3) correlation)
        │
        ▼
 [DATA] automatic Hough substitution          for z_rot==2 masters
        │                                     (m-3, 23, -43m, mmm, 222, mm2)
        ▼
 [DATA] map-wide variant unification          pseudo-symmetry classes unified
        │                                     per grain, render-NCC verified
        ▼
 [DATA] manual grain flip (optional)          user-driven per-grain correction
        │                                     in the Pattern-Match dialog
        ▼
 stored CrystalMap orientations  ◄════ everything below NEVER touches these
        │
        ▼
 [DISPLAY] IPF colouring                      standard colour key, or
        │                                     grain-consistent v2 (toggle)
        ▼
 [DISPLAY] per-phase filter                   one IPF map per phase + matching
        │                                     colour key (community standard)
        ▼
 layered canvas / PNG export
```

| # | Mechanism | Level | Trigger | Code |
|---|-----------|-------|---------|------|
| 1 | Hough substitution for unreliable masters | data | automatic in pipeline | `indexing_controller.py`; `spherical_unreliable()` in `backend/spherical_gpu/pseudosym.py` |
| 2 | Map-wide variant unification | data | automatic + "Unify variants" button | `backend/spherical_gpu/pipeline/variant_unification.py` |
| 3 | Render-verified small-grain adoption (stage 2.5) | data | part of unification | same file, `unify_map` |
| 4 | Manual grain flip | data | user, Pattern-Match dialog | `grain_snap_floodfill()` in `pseudosym.py`; `backend/api/routes/indexing.py` |
| 5 | Grain-consistent IPF colouring (v2) | display | "Stabilize colors per grain" toggle | `compute_ipf_colors_grain_consistent()` in `tools/phase_map_generator.py` |
| 6 | Per-phase IPF view + matching key | display | "Phase:" dropdown on IPF layers | `phase_filter` in `backend/api/routes/phase_map.py` |

---

## 3. Data-level mechanisms (they change stored orientations)

### 3.1 The pseudo-symmetry problem, precisely

EMsoft master patterns carry a `z_rot` factor. For crystals whose point group
is a **strict subgroup of the lattice holohedry** (e.g. alpha-AlFeMnSi: point
group m-3 on a cubic lattice whose holohedry is m-3m), the *band geometry* of
a Kikuchi pattern is (nearly) invariant under the **full holohedry**, but the
*intensities* are only invariant under the true crystal group. Consequences:

- **Variant classes.** The proper-rotation coset (holohedry / crystal group),
  taken at **Laue level** (Friedel's law — diffraction adds a centre of
  symmetry), splits orientations into *variant classes*: orientations in
  different classes produce **near-identical band geometry but different
  intensity distributions**. Counts: m-3 and 23 → **2 classes**; -3m → 2;
  4/m → 2; **-43m, 4mm, mm2 → 1 class** (their Laue groups equal the
  holohedry's Laue group — variants are *pattern-identical* and physically
  undecidable by any method; nothing needs fixing there).
- **The SHT correlation picks wrong variants.** The spherical cross-correlation
  scores the full pattern, but for these phases the score landscape has
  near-degenerate maxima at each variant; noise decides which one wins,
  pixel by pixel → *variant speckle*: a single grain indexed as a random mix
  of 40–80°-apart orientations.
- **Hough is variant-blind.** Hough indexing solves orientation from band
  *geometry* only, i.e. modulo the holohedry — it cannot distinguish variants
  either, but it is *consistent* (always the same representative), which makes
  it a stable geometric anchor.
- **The only variant-sensitive signal is the forward render.** Simulating the
  pattern for each candidate variant and correlating with the experiment
  (render-NCC) is the *only* discriminator. This was proven the hard way: a
  first resolver chose variants by minimum disorientation to Hough — 22% of
  its picks rendered as badly as the wrong variant. **Select variants by
  render-NCC, never by symmetry-distance.**

### 3.2 Automatic Hough substitution (`z_rot==2` gate)

For any phase whose master has `z_rot == 2` (covers cubic m-3 / 23 / -43m and
orthorhombic mmm / 222 / mm2), the pipeline replaces the spherical orientations
with **Hough orientations** outright (per-pixel fallback to the spherical
result only where Hough fails: `nmatch < 4` or `fit > 3°`). Rationale,
measured on a real 7050-alloy file: Hough renders NCC 0.69 vs. spherical
~90° off at 0.19, and indexes the whole 5304-px map in 4.4 s. The render-NCC
peak is very sharp (~2° FWHM), so no cheap refinement can rescue a
wrong-basin spherical seed.

The map records this: `metadata["orientation_source"] = "hough"`, and the UI
shows an "orientation from Hough" badge.

### 3.3 Map-wide variant unification

Because Hough is variant-blind, its output still mixes variant classes across
pixels. `unify_map()` (`backend/spherical_gpu/pipeline/variant_unification.py`)
fixes this **for the whole map at once** (no per-pixel clicking — infeasible on
large maps):

1. **Segment grains modulo the supergroup** (5° threshold). Under the
   holohedry all variants of one grain collapse into one segment — so the
   segmentation sees *physical* grains even when the stored orientations are a
   variant mix.
2. **Label each pixel's variant class** (proper coset representatives at Laue
   level; for metrically pseudo-cubic lattices the supergroup is detected from
   the lattice, `metric_supergroup_ops`).
3. **Coherence policy per grain:**
   - *Salt-and-pepper class labels* → mis-indexing → always unify to one class.
   - *≥ 2 spatially coherent domains* (each ≥ 8 px, minority ≥ 20%) → could be
     a real **pseudo-merohedral twin** → each domain verified independently,
     flipped only on a **clear render margin** (hysteresis 0.03); default keep.
     Twins are never destroyed silently.
4. **Decide the winning class by aggregated render-NCC** over the grain's best
   pixels (median over N best; OOM-hardened scorer: `torch.no_grad`, retry
   once after cache release, `-inf` never wins).
   *Ambiguity is best-vs-second margin* (< 0.01 → grain flagged ambiguous and
   left untouched, reported with centroid so the user can inspect).
5. **Stage 2.5 — render-verified small-grain adoption.** Generic catcher for
   *any systematic wrong basin that is NOT in the coset*: on the reference
   dataset, 20 small blobs sat ALL at an identical 71.9° (m-3) misorientation
   to the matrix — a second Hough band-coincidence basin, invisible to coset
   machinery (20 blobs sharing one exact misorientation are never real
   grains). A small grain (≤ 128 px) adjacent to a ≥ 3×-bigger donor is
   *tentatively* mapped onto the donor via the rigid correction
   C = mean(G)·mean(g)⁻¹ (left-multiplied) and **adopted only if the
   render-NCC margin is clear**. Real small grains render worse under
   adoption and stay bit-identical.
6. **Stage 3 — tiny-orphan rescue** (≤ 2 px islands joined to the surrounding
   grain, render-verified).

Runs automatically after indexing for affected phases and on demand via the
**"Unify variants (whole map)"** button (Phase Maps → Pseudo-Symmetry box).
The report lands in `metadata["variant_unification"]`.

**Evidence:** on a second reference dataset the true variant renders NCC 0.50
vs. the flipped variant 0.09 (margin 0.41 — the decision is not marginal).
After unification + adoption the exported `.ang` of the affected region had
**every** neighbour pair < 5° disorientation under m-3 — data provably smooth.

### 3.4 Manual grain flip (universal, any pseudo-symmetry)

For cases the automatics cannot know about, the Pattern-Match dialog (both on
the Indexing page and in Phase Maps) has a **variant gallery**: all candidate
orientations (current, crystallographic pseudo-variants via the holohedry
coset, Hough) rendered against the experimental pattern and ranked by
render-NCC; the user picks; **"Apply to grain"** flood-fills the connected
same-phase grain and *snaps each pixel individually* to the chosen variant's
branch (`grain_snap_floodfill`): per pixel it applies the coset operator
(LEFT multiplication h·q — crystal-frame symmetry acts from the left in the
Bunge/orix convention) that best matches the target, with parent-operator
tie-breaking and a 15° anti-drift cap, so **intra-grain gradients and
variant-split grains are handled correctly** (a rigid rotation of the whole
grain would corrupt intra-grain texture). One-level undo; optional guarded
Newton refine (only kept if converged, moved ≤ 2°, and the render-NCC sample
actually improved).

---

## 4. Display-level mechanisms (stored orientations untouched)

### 4.1 The IPF colour-key discontinuity — the residual "speckle" that was NOT data

After unification, the reference map still showed green jitter in IPF-Z. The
exported orientations were **provably smooth** — so the remaining artefact had
to be the colouring itself:

The standard TSL colour key (orix `IPFColorKeyTSL`) reduces each pixel's
sample direction into the **fundamental sector** of the phase's Laue group and
colours it by in-sector polar coordinates. For high symmetry (m-3m) the sector
is small and the key is effectively continuous. For **low-symmetry Laue groups
(m-3)** the sector is large and its boundary is a *discontinuity of the key*:
two crystallographically equivalent directions that fall on either side of the
boundary get **maximally different colours**.

**Measured on the real dataset** (814-px region, IPF-Z):
- a neighbour pair with only **1.29° misorientation** jumped blue↔green with
  RGB distance **1.34** (out of a max of √3 ≈ 1.73);
- **6.7%** of all sub-3° neighbour pairs jumped > 0.3 RGB;
- IPF-X and IPF-Y of the same data were clean — those directions happen to sit
  far from the sector boundary. (This asymmetry across X/Y/Z is itself the
  fingerprint of a key artefact rather than a data problem.)

**Literature grounding:** Nolze & Hielscher, *J. Appl. Cryst.* **49** (2016)
(IPF keys — continuity and uniqueness): only Laue classes **-4, -3 and -1**
fundamentally cannot have a colour key that is both unique and continuous;
for all others (including m-3) continuous keys exist. MTEX's documentation
carries the same "colour jumps" warning for its default keys. So the m-3
speckle is an artefact of the *standard key*, and fixing the display is
legitimate — the alternative (a globally smooth key) would change ALL
colours; Orienta chose a minimal, opt-in repair instead.

### 4.2 v1 — grain-mean stabilisation (superseded)

First iteration: colour every pixel by its **grain-mean** orientation. Kills
the speckle but **flattens real intra-grain gradients** (deformation colour
ramps disappear — measured correlation of colour distance vs. misorientation
≈ 0). Replaced by v2 under the same toggle.

### 4.3 v2 — grain-consistent branch colouring (current)

`compute_ipf_colors_grain_consistent()` in `tools/phase_map_generator.py`.
Principle: **colour every pixel by ITS OWN direction, but route all pixels of
a grain through the SAME side of the key discontinuity**:

1. Segment grains modulo the supergroup (same machinery as unification, 5°).
2. Reduce the grain-**mean** direction into the fundamental sector
   (standard path).
3. Per pixel: among its symmetry-equivalent directions pick the one **closest
   to the reduced grain-mean direction** (NOT the in-sector one).
4. Colour that representative with orix's **in-sector TSL colour math applied
   WITHOUT re-reduction** — `polar_coordinates_in_sector` +
   `rgb_from_polar_coordinates` from
   `orix/plot/direction_color_keys/_util.py`, bypassing
   `direction.in_fundamental_sector`. The formula extrapolates continuously for
   directions slightly outside the sector; RGB is clipped to [0, 1].
5. **Fallback:** any grain whose direction cluster spans > 10° from its mean
   uses the standard per-pixel path (extrapolation unvalidated that far out;
   such grains are rare and genuinely bent). For directions already in-sector
   the result is **bit-identical** to the standard key.

**Validation (real data, 812-px grain):** speckle pairs (< 3° misorientation,
> 0.3 RGB jump) **159 → 0**; intra-grain colour distance vs.
misorientation-to-grain-mean correlation **0.889** (v1: ~0 — flattened);
**729 distinct colours** in the grain (v1: 1). Toggle OFF stays byte-identical
to the untouched standard path (regression-tested in the development suite).

**Scientific integrity.** Is grain-consistent colouring "lying"? No: the IPF
colour of an orientation is a *class function* — all symmetry-equivalent
directions are equally "the" direction; the standard key also picks one
representative (the in-sector one), v2 merely picks a *grain-consistent*
representative instead of a per-pixel one. No orientation is altered, no
gradient is invented or removed; the choice is exactly as arbitrary as the
standard key's, only consistent. It is opt-in (default OFF), display-only,
and labelled as such in the tooltip. For publications, state: "IPF colours
use grain-consistent branch selection to avoid the m-3 colour-key
discontinuity (cf. Nolze & Hielscher 2016)."

### 4.4 Per-phase IPF view + matching colour key (community standard)

Mixing several phases in one IPF map with identical RGB codes is ambiguous —
**each phase has its own colour key (own fundamental sector)**, so the same
RGB means different directions in different phases. Community practice is one
IPF map per phase.

- `GET /api/phasemap/layer?...&phase_filter=<pid>`: on `ipf-x/y/z` layers,
  every phase except `<pid>` becomes transparent. Implemented as an
  **alpha-only mask applied AFTER colour computation** — the colour math is
  untouched, so it composes identically with the standard and the
  grain-stabilized path (`-1` = all phases, legacy behaviour, byte-identical).
- `GET /api/phasemap/ipf-key?...&phase_filter=<pid>`: the floating colour key
  shows **only that phase's triangle** — key and map always agree.
- UI: **"Phase:" dropdown** on every IPF layer (next to the stabilize
  checkbox), options from the same phase stats the legend uses; hidden for
  single-phase results. Typical use: add one IPF-Z layer per phase, filter
  each to a different phase, stack over a grey BC layer — or filter one layer
  and step through phases.
- **Export:** the composite PNG exporter captures layers as shown, so a
  phase-filtered stack exports exactly the community-standard single-phase
  IPF figure.

### 4.5 Related display fix: bbox-aware map clicks

Not colour-related but part of the same debugging arc: the layered canvas
auto-zooms to the non-transparent content bbox while the pointer math assumed
the full grid → clicks on visually correct pixels hit the wrong data ("Pixel
not indexed"). `pointerToRowCol` / `rowColToContainerPx`
(`frontend/src/components/EDS/mapCoords.js`) now compose **both** letterboxes
(container→canvas and canvas→bbox); EDS pages are byte-identical (optional
4th parameter).

---

## 5. What to use when (recipe)

Low-symmetry phase (m-3 approximant etc.) looks speckled in IPF:

1. **Nothing** — the pipeline already substituted Hough and unified variants
   automatically for `z_rot==2` phases. Check the map first.
2. Residual variant blobs / after manual edits → **Phase Maps →
   Pseudo-Symmetry → "Unify variants (whole map)"** (data-level,
   render-verified, twin-safe, undo via Pattern-Match dialog).
3. Remaining fine colour jitter on *smooth* data → **IPF layer →
   "Stabilize colors per grain"** (display-only v2).
4. One stubborn grain → **Pattern-Match dialog → "Fix pseudo-symmetry…"** →
   pick best-rendering variant → Apply to grain (+ optional refine, undo).
5. Multi-phase map → **IPF layer → "Phase:" dropdown**, one phase per layer /
   per export.

Note: colour toggles need no re-indexing — they re-render display layers only.

## 6. Known limits

- Chemically/structurally degenerate phases can win pixels of the wrong phase
  (e.g. a cubic approximant claiming matrix pixels). This is a *phase*-level
  problem, out of scope for the orientation work above; a render-verified
  phase-check feature is planned.
- `-43m` / `4mm` / `mm2` variants are pattern-identical (Laue-degenerate) —
  no method can or needs to resolve them.
