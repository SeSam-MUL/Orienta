# HDF5 Viewer

## What it does

The HDF5 Viewer (the "H5OINA Cockpit") is a low-level explorer for the contents
of an EBSD HDF5 / H5OINA file. Where the EBSD Viewer is focused on patterns and
preprocessing, the HDF5 Viewer lets you inspect **everything stored in the
file**: the raw HDF5 tree, all available maps (quality, Aztec indexing results,
geometry, EDS element maps, electron images), the diffraction pattern and EDS
spectrum at any pixel, and the file's metadata.

It is built around a 5-zone cockpit layout:

- **Top bar** — file open/close and inspection/scan tools.
- **Layer rail** (left) — every map layer the file offers, grouped by category
  (Quality, Indexing/Aztec, Geometry, EDS, Electron).
- **Map canvas** (centre) — the selected layer as a navigable 2-D map.
- **Pixel inspector** (right) — pattern, EDS spectrum, EDS counts, and stored
  Aztec result for the current pixel.
- **Bottom bar** — row/column and flat-index navigation, playback, and
  bookmarks.

It uses the backend `/api/h5/*` routes for opening files, fetching patterns/maps,
reading EDS data, walking the HDF5 tree, and running quality/defect scans.

## When to use it

Use the HDF5 Viewer when you want to **understand what is inside a file** rather
than process it. Typical situations:

- Checking which maps, EDS elements, and electron images a file contains before
  deciding how to analyse it.
- Inspecting the raw HDF5 group/dataset tree and attributes (for troubleshooting
  or verifying acquisition settings).
- Reviewing the vendor's (Aztec) pre-computed indexing/quality results stored in
  the file.
- Quickly scanning a scan for the best/worst patterns or for defects, then
  bookmarking pixels of interest.
- Handing the file off to the EBSD Viewer or the Indexing page (optionally with a
  pixel selection mask) to start the real workflow.

## How to use it — step by step

### Open and inspect

1. In the **top bar**, type or paste the file path (or click **Browse…** in the
   desktop app) and click **Open**. The path field turns green when a file is
   open. A file already opened in another module is picked up automatically.
2. Pick a map layer in the **Layer rail** on the left. Layers are grouped into
   Quality, Indexing (Aztec), Geometry, EDS, and Electron categories; only the
   categories the file actually contains are shown.
3. The chosen layer is drawn on the **map canvas**. Click a pixel on the map to
   select it; the right-hand inspector and the diffraction pattern update to that
   pixel.

### Inspect a pixel

4. In the **Pixel Inspector** (right), expand the panels:
   - **Pattern** — the diffraction pattern, switchable between **processed**,
     **raw**, and **background** sources, with live brightness/contrast/gamma/
     invert/equalise controls (display-only) plus **Auto** and **Reset**.
   - **Spectrum** — the full EDS spectrum at the pixel (if the file has EDS).
   - **Counts** — per-element EDS counts at the pixel.
   - **Aztec result** — the indexing result the vendor stored for that pixel.

### Navigate

5. Use the **bottom bar** to move: prev/next arrows, **Row** and **Col** number
   inputs, or the **flat-index slider/spinbox**. Keyboard arrows move by one
   pixel, Page Up/Down by a row, Home/End to the first/last pixel.
6. Press **Play** to auto-advance through the scan at the chosen frame rate
   (2–20 fps) — useful for visually scanning patterns.
7. **Bookmark** the current pixel to add it to the chip list at the bottom; click
   a chip to jump back. **Clear bookmarks** empties the list (bookmarks are saved
   per-file in local storage).

### Tools, scans, and hand-off

8. From the top bar, open **Summary**, **Metadata**, or **Tree** to see the
   file's overview, full metadata, and the raw HDF5 group/dataset hierarchy.
9. Run **Quality** or **Defect** scan to sweep the scan and rank pixels; the
   results open in a modal, and (for the quality scan) you can click a result to
   jump to that pixel.
10. **Export** saves the current pattern as a PNG (named with file/row/col/index).
11. **To Analysis** loads the file into the EBSD Viewer; **To Indexing** loads it
    into the Indexing page. If you have made a lasso/region selection on the map,
    that selection is carried over as a pixel mask so indexing can be restricted
    to those pixels.

## Inputs & outputs

- **Inputs:** an EBSD HDF5 / H5OINA file (Oxford, EDAX, Bruker). EDS maps,
  spectra, electron images, and Aztec indexing/quality results are read from the
  file if present.
- **In-session outputs:**
  - A hand-off of the file path to the EBSD Viewer or Indexing page.
  - A pixel-selection mask passed to Indexing (when a region is selected).
  - Per-file bookmarks (local storage).
- **File outputs:** **Export** writes the current pattern to a PNG. No bulk data
  export from this page.

## Tips & notes

- **This viewer reads the file; it does not modify it.** Pattern-source toggles
  and enhancement sliders only change what you see — they do not write anything
  back to the HDF5 file or to the indexing pipeline.
- **It shows the vendor's (Aztec) results, not Orienta's.** The Indexing category
  layers and the Aztec inspector panel display what the acquisition software
  already stored. Orienta does its own indexing in the Indexing module — these
  pre-rendered results are shown for reference only.
- **Selections carry to Indexing.** Lasso/region selection on the map becomes the
  pixel mask used by **To Indexing**, which is the way to index only part of a
  scan from here.
- **Quality/Defect scans subsample** the scan for speed and never raise on
  failure — an error is shown in red inside the results modal instead.
- **Large files load lazily.** Patterns and maps are fetched on demand, so very
  large scans open quickly and only the pixels you visit are read from disk.
- No GPU or WSL is involved in this module.
