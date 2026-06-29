# Crystal Hint

## What it does

Crystal Hint analyses experimental EBSD (Kikuchi) patterns directly to estimate
**which crystal phases could have produced them** — before, or independently of,
running a full indexing job. For a given pattern it:

- **Detects symmetry** from the pattern's n-fold rotational content and reports
  which crystal systems are compatible (cubic, hexagonal, tetragonal, …).
- **Estimates the lattice** (an approximate lattice-parameter range) from the
  detected Kikuchi-band geometry.
- **Ranks candidate phases** from your local library by how well their symmetry,
  lattice, chemistry, and (optionally) EDS composition fit the measured pattern.
- **Searches external crystallography databases** (the Crystallography Open
  Database, COD, and Materials Project) for matching phases that are *not* in
  your library, and lets you download a CIF for them.

It is deliberately complementary to indexing: indexing tells you the orientation
*assuming you already chose the right phase*; Crystal Hint helps you discover and
confirm *which phase* you are looking at.

## When to use it

Use Crystal Hint when:

- You have a pattern (or region) and are **not sure which phase it is** — for
  example an unknown intermetallic particle in an alloy.
- An indexing result looks suspicious and you want a **second opinion** on
  whether the indexed phase's symmetry actually matches the patterns.
- You suspect a phase is **missing from your SHT/CIF library** and want to find
  and import its structure.

It fits between loading data (in the EBSD Viewer / HDF5 Viewer) and building
your phase list for [Indexing]. You can feed its findings into the
[Crystal Database](CrystalDatabase.md) (download a CIF) and then
[Simulation](Simulation.md) (generate the master/SHT).

A dataset must be loaded first (either via the EBSD Viewer or the HDF5 Viewer);
the page reads patterns from the active session.

## How to use it — step by step

At the top of the page you set the shared context used by all modes:

- **Material preset** — pick an alloy family (e.g. an Al-alloy preset). The
  preset seeds the expected element list and a set of expected phases that get a
  ranking boost.
- **Chemistry (elements)** — the element symbols present in the sample. This is
  pre-filled from the preset and from any EDS-detected elements, and you can add
  or remove elements manually.

Then choose one of three modes via the tabs:

### Mode 1 — Single Pixel (deep analysis)

1. Pick a pixel. Type an index, or click on the **scan overview** thumbnail; the
   selection stays in sync with the EBSD Viewer / Phase Map.
2. Set options:
   - **Strict chemistry** — drop phases containing elements not in your sample.
   - **Neighborhood averaging** — `Single`, `3×3`, or `5×5`. Averaging
     neighbouring patterns suppresses noise and roughly doubles the detection
     confidence; `3×3` is the recommended default.
   - **EDS weighting** — `off` (pattern only), `soft` (down-weight phases whose
     composition disagrees with the pixel's EDS), or `filter` (drop them). It
     silently falls back to `off` if the dataset has no EDS.
3. Click **Analyse**. The right panel shows the pattern preview, the symmetry
   report, the lattice report, and a ranked **candidate list** (local matches
   plus external-database matches). If an indexing result is active, a badge
   shows whether the indexed phase's symmetry is consistent with what was
   detected.
4. For a promising external match, you can download its CIF into your library.

### Mode 2 — Region / Whole Scan

1. Choose the ROI: whole scan, a rectangle, or a polygon, and a cap on the
   number of pixels analysed (subsampled if the ROI is larger).
2. Click **Analyse**. You get **statistical histograms** across the region
   (n-fold distribution, crystal-system distribution, lattice-parameter
   distribution), the library-match rate, and a list of **missing-phase
   clusters** — recurring symmetry+lattice signatures that have *no* match in
   your library and are therefore prime candidates to add.

### Mode 3 — Quality Check

1. Requires an **active indexing result** (run Indexing first).
2. Choose the ROI and pixel cap as in Mode 2, then click **Analyse**.
3. The tool compares, per pixel, the indexed phase's crystal system against the
   symmetry detected from the raw pattern, and reports match / mismatch /
   unindexed counts plus a per-phase breakdown of likely mis-indexing.

## Inputs & outputs

**Inputs**

- The active EBSD dataset's patterns (it prefers the currently displayed,
  possibly processed, EBSD-Viewer signal; otherwise the raw file).
- Detector geometry (pattern centre, pattern size, beam energy) from the shared
  calibration — used for the spherical symmetry method and lattice estimation.
- Optional per-pixel EDS composition, the material preset, and the element list.
- For Mode 3, an active indexing result (CrystalMap).

**Outputs**

- On-screen reports only (symmetry, lattice, ranked candidates, histograms,
  cluster lists, quality-check statistics). Nothing is written automatically.
- A downloaded CIF (Mode 1 external matches) is the one persistent side effect —
  it is saved into `Database/CIF_Library/` for later conversion and simulation.

## Tips & notes

- **Symmetry detection is a heuristic, not a verdict.** On real, noisy raw
  patterns the rotational-NCC confidence is often `none`/`low`. When confidence
  is low the external search deliberately does *not* constrain by crystal system,
  so it can still surface low-symmetry intermetallics — but treat the
  crystal-system suggestion as a hint, not a fact.
- **Use neighborhood averaging.** `3×3` averaging materially improves detection
  on noisy patterns; prefer it for single-pixel analysis.
- **EDS weighting only works if the dataset carries EDS.** With EDAX/Ni or other
  EDS-free data it degrades to pattern-only ranking (a warning is shown).
- **External search needs a constraint.** A chemistry-free external query
  requires at least a crystal system or a lattice bound; otherwise it is refused
  rather than dumping the whole database.
- **Materials Project results need a configured API key.** Without one, the MP
  side returns nothing and only COD results are shown — this is silent and
  expected.
- **Downloading a CIF does not simulate it.** After import you still need to
  convert it ([Crystal Database](CrystalDatabase.md)) and generate the master /
  SHT ([Simulation](Simulation.md)) before you can index against it.
- **Pattern-degenerate phases are merged.** Phases with the same space group and
  near-identical lattice produce indistinguishable Kikuchi patterns, so they are
  collapsed into a single representative ("also matches: …") rather than each
  taking a top slot.

[Indexing]: ../README.md
