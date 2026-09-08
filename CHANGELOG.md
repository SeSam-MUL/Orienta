# Changelog

All notable changes to Orienta, newest first. While Orienta is pre-1.0, the
minor number (0.**x**.0) marks a release that changes how you work — a new
module, a reworked page — and the patch number (0.x.**y**) everything else:
fixes, and refinements to features that are already there.

You can see which version you are running under **Settings → About Orienta**.

---

## v0.2.6 — 2026-09-08

Nothing in the app behaves differently. This release is about what the project
says about itself — because from this version on, Orienta is public, and an
archived copy of it carries a DOI that people will cite.

### The README now matches the code

An audit compared every claim in the README against what is actually wired up,
and four of them were wrong.

- **"texture/ODF" was overstated.** The texture index and entropy come from
  texture-component volume-fraction bins, not from an orientation distribution
  function. Useful for comparing your own datasets against each other; not
  comparable with MTEX numbers. Said plainly now.
- **"Spherical (EMSphInx via WSL)" was out of date.** The default is the
  built-in GPU indexer; EMSphInx is the alternative, not the main path.
- **"EDAX H5" needed a limit.** Hexagonal-grid scans are not detected on the
  load path, so they open sheared instead of being refused. Until that is
  fixed, do not load HexGrid data — this is the one place where you can get a
  plausible-looking result that is wrong.
- **The status line was too pessimistic**, claiming parts of ML and refinement
  were hidden. Both are ordinary sidebar pages and have been for a long time.

There is now a **Known limitations** section carrying those, plus the ones
already known: the simulation engine cannot start because its automation module
is not part of this repository, the "Texture Components" map layer returns an
empty map, two recrystallisation statistics are placeholders, the frontend build
needs Node 20.19+, `requirements.txt` installs the CUDA runtime even without an
NVIDIA card, Python 3.12 is blocked by one leftover dependency, and a ZIP
download reports its version as "unknown".

### Attribution that was missing

- **EMsoft and SHTfile are now credited, with their licence texts.** Three files
  say of themselves that they carry transcribed EMsoft content — the WEKO
  elastic coefficients for Z = 1..98, the LFSR113 random-number stream, and two
  230-entry tables from the SHTfile reference implementation. Both upstreams are
  BSD-3-Clause and require their notice to travel with the source; neither
  licence text was in the repository. Both are now under `licenses/`.
- **The description of what ships was wrong in NOTICE.md.** It said runtime
  dependencies are "not redistributed", while three.js sits in
  `frontend/public/vendor/` and is loaded directly. It is now listed as
  redistributed, and vanta.js — whose minified bundle carries no copyright text
  at all — has the notice MIT requires beside it.
- Three more locations with derived code were undeclared, and one comment called
  kikuchipy MIT-licensed when it is GPL-3.0.

### The Materials Project key no longer lives in the source

It was a literal in `cif_database_builder.py`. It now comes from your own
settings, the same place the rest of the app already read it from, with the
`MP_API_KEY` environment variable as a fallback. Unconfigured means the online
lookups are skipped and the database builder works offline from your local CIF
files. **If you used that builder's online lookup, enter your own key under
Settings → API keys.**

### Citation metadata

`CITATION.cff` said version 0.1.0 and pointed at the DOI of that first archive.
It now tracks the current release and cites the **concept DOI**, which always
resolves to the newest archived version, and a `.zenodo.json` gives the archive
its full description and the funding acknowledgement.

---

## v0.2.5 — 2026-09-08

Figures in series, and a phase check that judges fairly. **If you have run
"Check phases" before, read the second section: its verdicts can differ now,
and the old ones were sometimes wrong.**

### Every element map as its own file

- **The EDS page can write one file per map.** Right-click a tile and choose
  *Export each map as its own file…*: the ordinary export dialog opens on the
  first map, you set resolution, border, format and scale bar once and see
  them on a real map, then one click writes the whole series. The folder is
  chosen once, not once per file.
- **Each file carries its own map's name** in the corner and in the file name,
  so a panel of element maps is not a set of pictures that all say the same
  thing. A switch in the dialog turns that off if you would rather caption the
  sample than the map.
- **Each file keeps its own micrometres per pixel.** The tiles do not share a
  raster — the electron images sit on the SEM raster and the element maps on
  the scan raster, on a real file 10.6x apart — so one shared number would
  mis-scale half the series.
- **The IPF colour key can be exported on its own**, on a white plate or
  transparent to lay onto a figure of your own. Right-click the key panel; it
  was the only picture on the phase map page without an export.

### Colour keys beside a map are no longer cut off

- **The border could never be wider than half the map.** A three-phase IPF key
  is 2.75x as tall as it is wide, and beside a wide, short map it needs about
  1.4 map heights above and below — so it was sliced across the top and the
  bottom, and the file came out at exactly twice the map height whatever you
  asked for. The border now takes the room the figure needs; the only limits
  left are what the browser can encode, and the dialog says when it had to
  trim.
- **The size the dialog announces is the size it writes.** It ignored a key
  that had grown since the border was last fitted.

### Checking phases got fairer — and it can change what you see

- **A phase is now judged at a fair orientation.** It used to be scored at its
  stored orientation while its rivals were scored at their best, which is not
  a comparison. Measured on a real map: **8 of 24 island findings were
  artefacts**, and the per-grain stage flipped a whole MgCuAl2 particle to Al
  (0.207 stored against 0.420 at a fair orientation).
- **The score no longer chases noise.** It compared a noisy measured pattern
  against a noise-free simulation across the full frequency range; limiting
  both sides to the band that carries the signal takes the median on a real
  7050 map from **0.196 to 0.417**. That is what had made every grain a
  suspect.
- **A second stage finds wrong pixels enclosed inside a grain** — islands
  below grain size (a grain being 9 pixels or more), taking the orientation
  from the grain around them. Outcomes are named: reassigned, rescued, or
  undecided — and undecided is never applied.

### Hough: the reflector budget is yours to set

- **A CIF written as P 1 asked for 44.7 GiB** for its band-triplet library and
  took the machine to its commit limit. Orienta now works out the cost before
  allocating anything, refuses what does not fit, and lets you set the number
  of reflector families per phase — on the Indexing page and in the phase
  verification panel. Your choice is remembered between sessions.
- **Trimming reflectors automatically was measured and rejected**: on Ni, 24
  families put 192 of 800 patterns 120 degrees off and 16 families gave a
  median fit of 180 degrees, silently. A refusal you can see beats a wrong
  answer you cannot.

### Fixes

- **The cleanup sliders and "Apply cleanup" now refresh the IPF and BC
  layers.** They had been showing the state before the cleanup.
- **A diagnostic layer that paints only its findings no longer decides how
  large the map is drawn.** A 39x136 scan collapsed onto the ~20 flagged
  pixels, drawn with a 100 nm scale bar instead of 2 um.
- **Pattern match refuses a render grid the graphics card cannot hold**
  instead of failing inside the driver.
- **The changelog listed v0.2.3 twice**, above v0.2.4.

---

## v0.2.4 — 2026-08-28

Scale bars, and a set of crashes that took a whole page down. If you exported
a figure from Orienta before today, please check its scale bar against this
release before the figure goes anywhere.

### Crashes that a single click could trigger

- **Switching to the Wand paint mode killed the entire EDS page.** One click
  was enough; no data needed to be loaded.
- **Opening the Phase Map from the Batch page killed the Phase Maps page**,
  every time — and retrying crashed again.
- **When a map export failed, the message about it crashed the page.** The
  notice meant to tell you what went wrong was the thing that went wrong.

### When the backend dies

- **The app now says so within seconds instead of appearing frozen.** Its
  health check shared the five-minute timeout used for long operations, so a
  backend struggling with memory — which keeps its connection open but answers
  nothing — left the app reporting "connected" while nothing worked. Measured
  against a backend that answers nothing: reported after 17 seconds, where
  before the first sign could not arrive for five minutes.
- **A batch run no longer spins forever** when the backend is gone; after three
  missed replies it says the connection was lost.
- **Running out of memory says so.** It used to end in "Indexing failed:" with
  no reason at all, because that particular error carries no message of its own.

### Saved results

- **A crash can no longer leave half a result file behind.** Results are now
  built alongside their final name and moved into place only once finished —
  before, a run killed midway left a file that opened cleanly, looked
  complete, and held only part of the data. A failed re-export also leaves the
  previous result intact instead of destroying it.

- **The scale bar on an exported image was wrong** wherever the picture was not
  the raster its measurement came from. On the test scans the error was between
  1.3x and 8.5x, and which one depended on the file. It affected the EDS
  overlay export, single-layer exports from the Phase Maps page, and the
  "all maps on one sheet" export. A bar labelled "2 um" could be 17 um long.
  Every export now takes its scale from the picture actually being written, so
  a map and an electron image exported from the same file carry their own
  correct bars.
- **A montage no longer offers a scale bar at all.** Its panels are scaled to a
  fixed cell width and separated by gaps, so no single bar can be true for the
  sheet.
- **The scale bar in the pattern-match figure works.** It could never show
  micrometres, and the pixel bar it fell back to drew one length while its
  label claimed another; dragging the bar's handles changed its length and left
  the label alone. Its length now follows the map it measures, and where a scan
  carries no step size it measures in map pixels instead of inventing a
  micrometre value.
- **The scale bar on screen follows the map after you resize the window.** It
  used to keep measuring against the width the map had before.
- **A map with no step size says so** instead of drawing a bar from a
  placeholder value of 1 um per pixel.
- **A step size of exactly 1 um is shown in the status bar again.** It was
  being hidden, on the assumption that it meant "unknown".
- **Right-clicking a single layer in the Phase Maps tile view offers that
  layer's export again.** It was offering the composed map instead.
- **The IPF-X and IPF-Y buttons on the EDS page work.** They were drawn,
  enabled and did nothing when pressed - which reads as a problem with your
  data rather than as an unconnected button.
- **"Download All" in the Database Browser now downloads what it says.** It
  opened a window headed "Sync Upload" that listed your *local* files and
  greyed out anything you did not already have - and then downloaded.
  Categories that exist only on the server could not be selected at all. The
  count shown after any completed sync, in either direction, was also missing.

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
