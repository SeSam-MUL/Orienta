# Changelog

All notable changes to Orienta, newest first. Versions follow
[semantic versioning](https://semver.org/): while Orienta is pre-1.0, the
minor number (0.**x**.0) marks a release with new features, and the patch
number (0.x.**y**) a fix-only release.

You can see which version you are running under **Settings → About Orienta**.

---

## v0.2.0 — 2026-08-27

The first release since v0.1.0, and a large one: 164 changes. The headline
items are a rebuilt EDS phase map, cropping a scan to a selection, and a
one-click problem report.

### Staying up to date

- **Orienta now checks for a new release when it starts** and offers to install
  it: it fetches the new version, installs any changed packages, rebuilds the
  interface and restarts. If any step fails, your working version stays exactly
  as it was.
- Only **released versions** are offered, never work in progress.
- You can skip a version, postpone the question, or switch the check off
  entirely under Settings → About Orienta.
- Installations set up from a downloaded archive (rather than a git clone)
  cannot update themselves; they are told so, with instructions, instead of
  being offered a button that cannot work.

### Reporting problems

- **Every install now has a version.** Settings → About Orienta shows it;
  quote it in bug reports.
- **"Report a problem…"** in Settings and on the crash screen packages your
  description, the app version, environment details and the log files into one
  zip to attach. It contains no measurement data and is never sent
  automatically. See [the guide](docs/user-guide/ReportingProblems.md).
- **Orienta now keeps log files** (`logs/`), so the cause of a failure survives
  closing the window — including crashes that happen before the interface even
  appears.

### EDS phase map

- **The map is grouped into regions by chemistry first**, and phases are
  assigned to those regions afterwards — instead of deciding pixel by pixel.
  Regions can be merged, split, grown, shrunk and snapped to chemical edges.
- **Region inspector** below the map: full composition with spread, enrichment
  against the map background, neighbours, and candidate phases with their
  deviation.
- **Magic-wand selection** — click a feature to select all of it, with live
  preview, a readout of what is selected, undo, and map-wide phase replacement.
- **Rules that decide which phases may compete** for a region (e.g. "silicon at
  least twice the background"), so ambiguous EDS signals stop pulling the wrong
  phase in.
- **Phase colours you pick**, shared with the EBSD phase map, and an overview
  that folds many regions into few phases.
- The phase map has **its own tab**, can be **zoomed with the scroll wheel**,
  **overlaid on an element map or the SEM image**, and **exported by right-click**.
- **EDS-only files** (Aztec element-distribution exports without EBSD data) now
  open directly on the EDS page.

### Cropping a scan

- **Crop to a rectangle, ellipse or lasso selection** in the EBSD viewer —
  patterns, EDS maps, band contrast and per-pixel data all follow, without loss.
- **Export the crop as a standalone file** (a 546 MB source became a 42 MB file
  in testing, patterns bit-identical).
- Indexing, including batch runs, honours a non-rectangular selection.
- Electron images follow the crop window through their own micrometre geometry.

### Indexing & phase identification

- **Render-verified phase check and grain reassignment** — find and correct
  regions where a phase was assigned that the pattern does not support.
- **Pseudo-symmetry tools**: map-wide variant unification, grain flip with coset
  snap and undo, a variant gallery with neighbour-grain and reference-pixel
  candidates.
- **Grain-stabilised IPF colouring** removes the speckle caused by colour-key
  discontinuities, and IPF maps can be **filtered to a single phase**.
- **EDAX .up1/.up2 and .osc files** can be loaded.
- Hough indexing works for CIFs written in a non-standard space-group setting
  (previously it tried to allocate over a terabyte and failed).
- Spherical indexing falls back to the CPU on machines without an NVIDIA GPU
  instead of crashing.
- Indexing speed is reported as patterns/second for **all three methods**, so
  they can be compared.

### Maps, images and export

- **Image export everywhere** — right-click a pattern, map or preview for a
  dialog with crop, resolution, format, border, scale bar and caption.
- **Scale bars and colour keys are movable objects** in the exported figure,
  not fixed columns.
- **Grain boundaries** as a map layer, classified by misorientation into three
  freely adjustable bands.
- **Layer picker with search** that only offers layers which can actually be
  drawn, and says what is missing for the rest.
- **Value-scale legends** for every layer whose colours mean a number.
- **3D crystal-structure viewer** for CIF and XTAL files in the database browser.

### Fixes worth knowing about

- **Closing the app now really stops the backend.** A surviving backend used to
  hold port 8000 and, after a drive was unplugged, served dead file handles —
  every file load then failed with a cryptic error.
- Band contrast comes from a single service with a clear order of preference
  (native → stored → computed) and is named accordingly, instead of silently
  substituting a different measure.
- Saved results re-import with the geometry they were computed with, so a
  pattern match after reloading matches again.
- Batch indexing with Hough or Dictionary no longer crashes at startup.

### New in the interface

- New wordmark, logo and favicon.
- The About section carries the licence notice, the version and the problem
  report button.

---

## v0.1.0 — 2026-06-29

Initial release for beta testers.
