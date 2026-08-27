# Changelog

All notable changes to Orienta, newest first. While Orienta is pre-1.0, the
minor number (0.**x**.0) marks a release that changes how you work — a new
module, a reworked page — and the patch number (0.x.**y**) everything else:
fixes, and refinements to features that are already there.

You can see which version you are running under **Settings → About Orienta**.

---

## v0.2.3 — 2026-08-27

Fixes for four things users reported, three of which read as "it just does
nothing".

- **The Phase Maps page crashed every time it was opened.** It has been
  unusable since v0.2.0 — three separate reports were this one fault.
- **Detector width no longer starts at an impossible value.** It was preset to
  0.1 mm, which is not a detector at any pattern size; that value quietly
  skewed the pattern centre and everything computed from it. It is now read
  from the measurement file, and where the file does not carry it, estimated
  and clearly labelled as an estimate. A value no real detector could have is
  now called out — this matters because a wrong detector geometry leaves the
  maps looking perfectly correct.
- **Choosing a map layer now shows that layer.** Until now the map only
  reloaded when you pressed "Refresh", so picking "Grain boundaries" appeared
  to do nothing at all. When a layer cannot be drawn yet, the reason is shown
  as a warning instead of a quiet status line.
- **IPF maps no longer offer a colormap.** IPF colours are a fixed code for
  crystal directions, not a value range; the setting never had any effect on
  them and only suggested otherwise. Colormaps remain where they belong, on
  Band Contrast, KAM, GOS and the other scalar maps.
- The status bar shows the grid size as "150×201" again, instead of the
  raw escape sequence it had been printing.
- "Load CIF" no longer fails in browser mode.

---

## v0.2.2 — 2026-08-27

Problem reports become easier to file and easier for us to act on.

- **"⚠ Report a problem" is now in the toolbar at the top of every page.** You
  no longer have to leave the page where something went wrong in order to
  report it — which also means the record of what happened is more complete.
- **A picture of the window is included automatically** (desktop app only),
  taken the moment you open the dialog, so you no longer need to make a
  screenshot yourself. You can leave it out if the screen shows something you
  would rather not share.
- **Every report carries a short error id.** The same fault produces the same
  id on every machine, so if two people run into one problem it is visible as
  one problem instead of two unrelated reports. It is shown in the dialog and
  written into the report.
- **"Open a GitHub issue"** opens a new issue with your description, the
  version, the error id and the preceding events already filled in — you only
  drag the report file into it. (GitHub does not allow programs to attach
  files to an issue, so that last step stays manual.)

---

## v0.2.1 — 2026-08-27

Fixes to the update check itself, found by running it against a real
installation.

- **Says when it cannot sign in.** A private repository answers "not found"
  rather than "access denied" when credentials are missing, so the check used
  to report "server unreachable" and send you looking for a network problem.
  It now tells you to sign in once instead.
- **The check never opens a credential window on its own.** It runs in the
  background at start-up, where a prompt nobody asked for would either hang or
  appear out of nowhere.
- **"Check for updates" now says what it found** — up to date, not signed in,
  or unreachable — instead of only ever reporting "up to date".
- Release notes with dashes and umlauts are no longer garbled on Windows.

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
