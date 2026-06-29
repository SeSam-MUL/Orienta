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

### Process patterns

8. In **Pattern Processing**, apply (in any order): **BG Dynamic** / **BG
   Static** (background removal), **Frame Average** (with a window-size input),
   **Auto-Contrast**, or **CLAHE** (adaptive histogram equalisation, with a
   kernel-size input). Use **Recommended Pipeline** to run Static BG → Dynamic BG
   → CLAHE in one click.
9. With **Auto-deepcopy** ticked (default), each destructive step is applied to a
   new derived dataset (e.g. `dataset_bg_dyn`) so your original stays intact. The
   **Datasets** list lets you switch between the original and derived versions,
   compare two side by side, or remove a derived dataset.
10. The **Brightness**, **Filter** (Sobel/Canny/Sharpen/…), and **Interpolation**
    controls are display-only adjustments to the on-screen pattern (they do not
    alter the stored data).

### Detector mask, EDS, and PC hand-off

11. **Detector Signal Mask** applies a circular aperture. It is auto-enabled for
    EDAX-style files (whose corners are blank) and off for Oxford H5OINA (whose
    full rectangular frame carries real signal). Adjust the radius slider if
    needed.
12. **EDS Composition** (if the file has EDS) shows the element composition at the
    current pixel; switch the unit between **Counts / Wt.% / At.%** and optionally
    overlay the top elements on the pattern. **Send to Indexing** forwards a
    chemistry-based pixel filter to the Indexing page.
13. To calibrate the pattern centre, navigate to a good pattern and click
    **+ Add current pattern to PC Refinement** (in the highlighted PC Refinement
    box or the pattern action footer). The first add jumps you to the PC
    Refinement page; the badge counts how many patterns you have collected.

## Inputs & outputs

- **Inputs:** an Oxford H5OINA or EDAX/Bruker HDF5 EBSD file. Optional EDS data
  inside the same file enables the composition panel.
- **In-session outputs:**
  - The loaded EBSD signal and any derived (processed) datasets, held in backend
    memory for use by Indexing, PC Refinement, EDS, and Phase Maps.
  - The list of patterns sent to PC Refinement.
  - A chemistry mask forwarded to Indexing (when you use **Send to Indexing**).
- **File outputs:** **Export Pattern** saves the currently displayed pattern as a
  PNG. (There is no scan-data export from this page.)

## Tips & notes

- **Vendor matters for the mask.** Leave the detector mask off for Oxford H5OINA
  — enabling it would discard real diffraction data in the corners. It is
  correctly on by default for EDAX-style files.
- **Overview default is Band Contrast.** Aztec's stored Band-Contrast array reads
  in milliseconds regardless of file size. The intensity-based modes (Mean, etc.)
  must read patterns from disk; on very large lazy-loaded scans they are sampled
  (down-strided) and marked as a "sampled preview", then upscaled.
- **Processing is destructive but safe by default.** Keep **Auto-deepcopy**
  ticked so each step writes to a derived dataset and your raw data is preserved.
- **CLAHE shows real progress.** Long CLAHE runs report a progress bar with
  percentage and ETA; other operations show a simple spinner.
- **Restarting the backend loses loaded data.** The signal lives in memory only;
  after a restart you must re-load the file (the Recent Files list helps).
- The **Merge** button in the Datasets group is currently a placeholder and does
  not yet combine datasets.
