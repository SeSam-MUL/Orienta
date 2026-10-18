# EBSD Viewer

## What it does

The EBSD Viewer is the main entry point for loading an EBSD scan and inspecting
its Kikuchi diffraction patterns. It loads an Oxford H5OINA or EDAX/Bruker HDF5
dataset, shows a navigable **overview map** of pattern quality across the scan on
the left and the **diffraction pattern** at the selected pixel on the right, and
provides the pattern-preprocessing steps (background removal, frame averaging,
contrast enhancement) that prepare patterns for indexing.

It also surfaces per-pixel **EDS composition** (when the file contains EDS data)
and acts as the hand-off point into **PC Refinement** — you collect good patterns
here and send them on for pattern-centre calibration.

Internally it talks to the backend `/api/ebsd/*` routes for loading, pattern
fetching, the overview reductions, and all processing operations.

## When to use it

This is normally the **first** module you open in a session. Use it to:

- Load an EBSD file and confirm the grid dimensions, pattern shape, and vendor.
- Browse the scan, find high-quality (well-diffracting) regions, and judge data
  quality before committing to indexing.
- Clean up patterns (subtract background, average frames, enhance contrast) so
  downstream indexing and PC refinement get good input.
- Pick representative patterns and add them to the PC Refinement list.

## How to use it — step by step

### Load a file

1. In the left panel's **Load Data** group, type or paste the file path, or
   click **Browse…** to pick a file, or simply **drag-and-drop** an
   `.h5oina` / `.h5` / `.hdf5` file onto the page.
2. Click **Load Data** (or press Enter). A progress modal shows the load stages
   (reading metadata → building signal → detecting features → finalising). Large
   lazy-loaded files become navigable within seconds; the overview map fills in
   afterwards in the background.
3. The header **File switcher** and the **Loaded Files** list (left panel, shown
   when more than one file is loaded) let you switch between several files loaded
   this session. Switching re-loads the chosen file from disk.

### Navigate and inspect

4. Click anywhere on the **overview map** (left) to jump to that pixel, or
   **drag** across it to scrub through patterns (a fast thumbnail is shown while
   dragging). The selected pixel pattern appears on the right.
5. Use the arrow buttons, the **Row/Col** number inputs + **Go**, or the keyboard
   arrow keys (hold Shift to step by 10) to move. **Best** / **Worst** /
   **Random** buttons jump to the highest/lowest-quality pixel or a random one.
6. Change the **overview mode** with the dropdown or the quick-mode buttons
   (σ, BC, ∇, SNR, NCC, H) to view different quality metrics: Mean/Max Intensity,
   Std-Dev, Band Contrast, Sharpness (Laplacian), SNR (MAD), Neighbour
   Correlation, or Entropy. Enable the **quality mask** + threshold to grey out
   low-quality pixels.
7. Zoom the pattern with the mouse wheel and pan by dragging; press `R` or
   double-click to reset.

### Crop to a selection

Cropping cuts the loaded scan down to a region you draw, as a new dataset. Do it
*before* preprocessing and indexing: on a large file that turns a background
removal from minutes into seconds, and an indexing run from hours into minutes.
Nothing is lost — the original dataset stays in the **Datasets** list.

8. In the **Selection** bar above the overview, click **Rectangle**, **Ellipse**
   or **Lasso**. The tools are toggles: while one is armed a plain drag draws the
   shape and clicking no longer jumps to a pixel. Press the armed tool again to
   turn it off and get click-to-navigate back. With no tool armed you can still
   draw with **Shift+drag**, and while a tool is armed **Ctrl+drag** pans a
   zoomed overview.
9. Drag on the overview. The line under it reports how many pixels the selection
   covers and roughly how much memory the crop will need — for an ellipse or a
   lasso it counts the pixels *inside the shape*, not the box around it.
10. Click **Crop to selection**. The cut-out appears in the **Datasets** list and
    becomes active, the overview redraws to it, and the selection tool disarms so
    the next click navigates. Everything downstream now works on the cut-out.
11. To keep it, click **Save crop…** and choose a filename. That writes a
    standalone measurement file — patterns as they are *now* (with any background
    removal already applied), the EDS maps, Band Contrast and electron images —
    which you can load again like any other file.

**What follows a crop, and what does not.** EDS element maps, Band Contrast,
electron images, the phase and IPF layers, hover probes, line scans and region
statistics all show the cut-out. Indexing with **Hough**, **Dictionary** and
**Spherical (GPU)** runs on it, and an ellipse or lasso restricts indexing to the
pixels you drew. **Spherical with the EMSphInx (CPU) backend refuses a cropped
dataset** and says so — it cannot address a subset of the file, and would
otherwise index the wrong pixels. Cropping a crop is allowed and composes: the
result is recorded relative to the original scan, not to the intermediate.

### Process patterns

12. In **Pattern Processing**, apply (in any order): **BG Dynamic** / **BG
    Static** (background removal), **Frame Average** (with a window-size input),
    **Auto-Contrast**, or **CLAHE** (adaptive histogram equalisation, with a
    kernel-size input). Use **Recommended Pipeline** to run Static BG → Dynamic BG
    → CLAHE in one click. For very noisy scans there is also **NLPAR denoise**
    (see below).
13. With **Auto-deepcopy** ticked (default), each destructive step is applied to a
    new derived dataset (e.g. `dataset_bg_dyn`) so your original stays intact. The
    **Datasets** list lets you switch between the original and derived versions,
    compare two side by side, or remove a derived dataset.
14. The **Brightness**, **Filter** (Sobel/Canny/Sharpen/…), and **Interpolation**
    controls are display-only adjustments to the on-screen pattern (they do not
    alter the stored data).

### Detector mask, EDS, and PC hand-off

15. **Detector Signal Mask** applies a circular aperture. It is auto-enabled for
    EDAX-style files (whose corners are blank) and off for Oxford H5OINA (whose
    full rectangular frame carries real signal). Adjust the radius slider if
    needed.
16. **EDS Composition** (if the file has EDS) shows the element composition at the
    current pixel; switch the unit between **Counts / Wt.% / At.%** and optionally
    overlay the top elements on the pattern. **Send to Indexing** forwards a
    chemistry-based pixel filter to the Indexing page.
17. To calibrate the pattern centre, navigate to a good pattern and click
    **+ Add current pattern to PC Refinement** (in the highlighted PC Refinement
    box or the pattern action footer). The first add jumps you to the PC
    Refinement page; the badge counts how many patterns you have collected.

## Inputs & outputs

- **Inputs:** an Oxford H5OINA or EDAX/Bruker HDF5 EBSD file. Optional EDS data
  inside the same file enables the composition panel.
- **In-session outputs:**
  - The loaded EBSD signal and any derived (processed or cropped) datasets, held
    in backend memory for use by Indexing, PC Refinement, EDS, and Phase Maps.
  - The list of patterns sent to PC Refinement.
  - A chemistry mask forwarded to Indexing (when you use **Send to Indexing**).
- **File outputs:**
  - **Export Pattern** saves the currently displayed pattern as a PNG.
  - **Save crop…** writes the active cut-out as a standalone measurement file
    (patterns in their current, processed state, plus EDS, Band Contrast and
    electron images), which loads again like any other file.

### Denoise a very noisy scan (NLPAR)

**NLPAR denoise** replaces every pattern by a weighted average of similar
patterns in its neighbourhood (non-local pattern averaging, as implemented in
PyEBSDIndex). Patterns of the same grain are averaged, patterns across a grain
boundary get little weight, and the weights come from the noise measured in the
scan itself, so there is no strength to set. The one setting is the **NLPAR
radius** (1–3): how many neighbouring patterns in each direction may contribute.
A larger radius averages more and takes longer.

- **When it helps.** When indexing the raw patterns fails because the scan is
  very noisy (short exposure, high detector gain). On a scan that already indexes
  well it brings nothing and can spoil grain-boundary pixels: on a clean nickel
  scan, dictionary indexing lost 1 percentage point of pixels within 2° of the
  reference, all of them at grain boundaries. **Frame Average** is the cheaper
  alternative.
- **Radius.** 3 (the default) was the best on the very noisy scans below. If the
  grains or particles of your sample are only a few pixels across, the 7 × 7
  window reaches across their boundaries; use 1 or 2 there (on an aluminium alloy
  with fine grains, Hough indexing lost 0.7 points at radius 1 and 1.9 at radius
  3).
- **Advice.** Under the radius field the panel says whether NLPAR is likely to
  help on the active scan. It estimates, from about 1000 patterns (about a second
  at most), how much of the difference between neighbouring patterns is noise:
  where that is most of it ("Recommended"), NLPAR may help; where the
  neighbours differ much more than noise explains ("Not recommended": a clean
  scan, or structure finer than the search window), it may blur fine structures
  such as grain boundaries. The numbers are shown (strength, noise level in
  counts). It is an estimate, not a prediction: the limit (a strength of 1.7) was
  set on a few measured test scans (the nickel series below, a clean nickel scan
  that lost 1.0 points and an aluminium alloy with fine grains that lost 1.9 to
  6.6 points, both "not recommended"), using raw patterns. The estimate is made on
  the dataset as it is now, so after background removal or CLAHE it can differ.
  You can still run NLPAR, on a copy. A dataset that NLPAR has already been applied to
  says so and its button is disabled, as it is for a scan NLPAR cannot run on
  (too small, hexagonal, uniform); choose the original dataset to run it again.
  The Recommended Pipeline's "denoise first" is skipped (with a log line) in
  those cases.
- **Order.** Run it first, on the raw patterns, before any background removal:
  the noise estimate and the protection of saturated pixels assume raw detector
  counts. Tick **Denoise first (NLPAR) in the Recommended Pipeline** to have the
  pipeline do this for you; it is off by default. Oxford files hold patterns that
  the acquisition software has already processed; NLPAR on those has not been
  characterised.
- **Where the result goes.** Like the other steps it is applied to a derived
  dataset when **Auto-deepcopy** is ticked (`<name>_nlpar`), and every indexing
  method and the export use the denoised patterns. A dataset that was already
  denoised is refused, so the same noise is not averaged away twice.
- **Size limit.** The denoised patterns are a second copy in memory, so a scan
  that is too large for that is refused with a message; crop it first (see
  *Crop to a selection*). EDAX hexagonal-grid scans are not supported. It runs
  on the CPU.
- **Measured.** Public nickel gain series (Anes et al., Zenodo 7497682; 29 800
  patterns of 60 × 60 px per scan), dictionary indexing on the GPU against the
  orientations of the 0 dB scan of the same area, share of pixels within 2°:

  | noise | raw | NLPAR, radius 3 | radius 2 | radius 1 |
  |---|---|---|---|---|
  | 17 dB | 0.68 | 0.85 | 0.85 | 0.83 |
  | 24 dB | 0.09 | 0.80 | 0.77 | 0.64 |

  The whole map takes about 2 s on a 24-thread CPU (60 × 60 patterns, radius 3);
  expect longer on a laptop.
- **Cancel.** The **Cancel** button in the progress overlay stops the run at the
  next block of rows and leaves the dataset exactly as it was. The first run in a
  session spends some seconds preparing the numerical kernels; the overlay says so.

## Tips & notes

- **Crop first, then preprocess.** The point of cropping is that everything after
  it is small: the cut-out is pulled into memory (up to 4 GiB — above that it
  stays lazy and the log says so), so a background removal or an indexing run
  costs a fraction of what the full scan would. Cropping *after* a long
  preprocessing run wastes that run on pixels you then throw away.
- **A saved lasso comes back as its bounding rectangle.** The drawn shape is
  recorded in the exported file, but reloading it does not restore the selection
  — you get the rectangle around it, with every pattern present. Nothing is lost
  or wrong; you would redraw the shape to restrict indexing again.
- **Electron images can decline to follow a crop.** They sit on their own, finer
  pixel grid, so the crop travels to them through physical micrometres. If the
  file does not state both areas' step size and origin — or states origins that
  disagree — the image is shown **whole** with a warning chip in the layer row,
  rather than being cut to a guessed position. A wrongly placed cut-out would
  misplace every feature on it.
- **Vendor matters for the mask.** Leave the detector mask off for Oxford H5OINA
  — enabling it would discard real diffraction data in the corners. It is
  correctly on by default for EDAX-style files.
- **Overview default is Band Contrast.** Aztec's stored Band-Contrast array reads
  in milliseconds regardless of file size. The intensity-based modes (Mean, etc.)
  must read patterns from disk; on very large lazy-loaded scans they are sampled
  (down-strided) and marked as a "sampled preview", then upscaled.
- **Band Contrast is native when the file has it, computed otherwise.** In the
  **Band Contrast** overview mode Orienta prefers the vendor's real per-pixel
  quality: when the file carries Oxford's stored Band Contrast it is read straight
  from the file and shown as **"Band Contrast (native)"**. When it does not (EDAX,
  synthetic data, some imported results), Orienta instead computes a kikuchipy
  FFT pattern-quality per pixel and shows it as **"Pattern Quality (computed)"** —
  a genuine sharpness-of-diffraction proxy, not a stand-in for Band Contrast. A
  small **provenance chip** in the overview caption tells you which of the two you
  are currently looking at, so you never mistake a computed proxy for a measured
  value. This distinction matters downstream: the Band-Contrast analyses in the
  Analysis module (GMM, quality filter) only accept the native value (see
  `Analysis.md`).
- **Processing is destructive but safe by default.** Keep **Auto-deepcopy**
  ticked so each step writes to a derived dataset and your raw data is preserved.
- **CLAHE shows real progress.** Long CLAHE runs report a progress bar with
  percentage and ETA; other operations show a simple spinner.
- **Restarting the backend loses loaded data.** The signal lives in memory only;
  after a restart you must re-load the file (the Recent Files list helps).
- The **Merge** button in the Datasets group is currently a placeholder and does
  not yet combine datasets.
