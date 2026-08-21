# Indexing

## What it does

The Indexing page assigns a **crystal phase and orientation** to each pixel of a
loaded EBSD scan by matching its Kikuchi pattern against candidate phases. It is
the heart of the workflow: its output (a per-pixel orientation/phase map, the
*CrystalMap*) feeds the [Phase Map](PhaseMap.md), [Pole Figure](PoleFigure.md),
and EBSD Analysis tools.

Three indexing methods are available:

- **Hough indexing** — the standard band-detection approach. Needs a **CIF** file
  per phase. Fast, good for cubic/well-resolved patterns.
- **Dictionary indexing** — correlates each experimental pattern against a
  dictionary of simulated patterns built from an EMsoft **master `.h5`** file.
  Robust, supports a GPU path.
- **Spherical indexing** — spherical-harmonic (SO(3)) correlation against an
  EMsoft **`.sht`** master. Needs a `.sht` file per phase; runs either on the
  in-process GPU backend or via EMSphInx (CPU, in WSL).

You choose **which pixels** to index (whole image, a rectangular region, or a
chemistry-based mask) and **which phases** to test. The backend keeps the original
pixel positions intact when you index a subset, so results map back onto the full
scan correctly.

The page talks to the backend `/api/indexing/*` routes for starting jobs, polling
progress, retrieving results, pattern-match inspection, and export.

## When to use it

Use Indexing **after** you have loaded a scan in the [EBSD Viewer](EBSDViewer.md)
and (ideally) calibrated the pattern centre in [PC Refinement](PCRefinement.md).
A good pattern centre is critical — indexing quality stands or falls with it.

Typical entry points:

- You know your candidate phases and want a full phase/orientation map.
- You used **Send to Indexing** from the EDS page and want to index only pixels of
  a given chemistry, or route each phase to its own chemically-classified pixels.
- You used the **Phase Test** to identify an unknown phase and clicked
  *Use for indexing*.

## How to use it — step by step

### 1. Pick the dataset and method

1. Confirm the active dataset in the **Dataset** dropdown at the top (it lists
   loaded files and any derived/processed datasets). The green/red **data status**
   line tells you whether EBSD data is loaded.
2. Choose the **Method** dropdown: *Hough*, *Dictionary*, or *Spherical*.
   (An *Embedding* option is listed but **not implemented** — see Tips.)

### 2. Add phases

3. In **Phase Selection**, click **+ Add** to open the floating phase picker. Only
   files matching the current method are offered (CIF for Hough, master `.h5` for
   Dictionary, `.sht` for Spherical). Selected phases appear in the list; remove
   one with its × button.
4. If the app flags **degenerate** phases (entries EBSD cannot tell apart), use
   **Reduce phases** to trim the list — fewer near-identical candidates means a
   cleaner result.

### 3. Set method parameters

5. **Hough:** set *Bands*, *t-sigma*, *r-sigma* (band-detection sensitivity).
6. **Dictionary:** set *Metric* (NCC/…), *Keep N* (top matches retained),
   *Resolution* (dictionary angular step, degrees) and *Energy* (kV). Use
   **Generate Dictionary…** if you need to build the dictionary first. Pick the
   **Compute** mode — *Auto*, *GPU*, or *CPU* (GPU is disabled when no CUDA device
   is detected, and shows the detected card + free VRAM).
7. **Spherical:** pick the **Backend** (*Spherical GPU* in-process, or *EMSphInx*
   CPU via WSL), the **Bandwidth** (higher = sharper, more VRAM), **Regions**,
   the **Refine** checkbox (orientation refinement after the coarse search),
   **Gaussian background**, and the **Circular mask** option. A live
   **preprocessing preview** shows the effect of these settings on a sample
   pattern.

### 4. Choose which pixels to index

8. In the left panel's **Pixel Selection** group choose **Full image**,
   **Region**, or **Chemistry mask**.
   - *Region:* enter row/column ranges, or add several saved regions.
   - *Chemistry mask:* build element filters (e.g. Fe at% > 0.3), combine them with
     AND/OR, and apply. When a phase map was handed over from EDS, a
     **Per-phase routing** toggle appears so each phase is indexed only on its own
     classified pixels.

### 5. Run and monitor

9. Watch the **Runtime banner** — it shows whether the run will use CUDA (and how
   much VRAM is free) or fall back to CPU. If it is red ("low VRAM"), click
   **Release GPU** before starting.
10. Click **Start Indexing**. Progress, an elapsed timer, and a live log appear.
    Use **Stop** to cancel. Keep **Send to Phase Map** ticked to make the result
    the active map automatically.

### 6. Inspect, refine, export

11. After completion a green quality summary appears. For multi-phase runs,
    **Details** opens a per-phase breakdown (pixel counts, mean CI).
12. **View Pattern Matches** opens the [Pattern Match](PatternMatch.md) inspector
    (experimental vs. simulated pattern + NCC, per pixel). **Refine** (Dictionary
    results) runs Nelder-Mead orientation refinement.
13. **Phase Test** runs the single-pixel phase identification dialog. **Batch**
    indexes multiple files. **→ Analysis** hands the result to the EBSD Analysis
    module.
14. Use the **Export** dropdown to save the result as `.h5` (rich/light) or `.ang`.

## Inputs & outputs

- **Inputs:**
  - A loaded EBSD scan (from the EBSD Viewer).
  - Phase files matching the method: **CIF** (Hough), **master `.h5`**
    (Dictionary), **`.sht`** (Spherical).
  - A detector / pattern centre (from PC Refinement; otherwise a fallback detector
    is built from the signal shape).
  - Optional: a chemistry mask or an EDS phase-map routing handoff.
- **Outputs:**
  - A per-pixel **CrystalMap** (phase id + orientation + confidence) stored in
    backend memory and, by default, set as the active result for the Phase Map /
    Pole Figure / Analysis tools.
  - Exported result files: `.h5` (rich or light) and `.ang`.

## Tips & notes

- **Cropping restricts what gets indexed.** If you cut the dataset down in the
  EBSD Viewer (see `EBSDViewer.md`), indexing runs on the cut-out, and an
  ellipse or lasso restricts it further to the pixels you drew — the ones
  outside keep their patterns but are left unindexed. The result records where
  in the original scan it came from. **Hough**, **Dictionary** and **Spherical
  (GPU)** all honour this; **Spherical with the EMSphInx (CPU) backend refuses a
  cropped dataset** rather than silently indexing the wrong region of the file.
- **The pattern centre dominates quality.** Calibrate it in PC Refinement first;
  an uncalibrated PC is the most common cause of poor or wrong indexing.
- **GPU vs. CPU.** Dictionary and Spherical have GPU paths (PyTorch/CUDA). The
  in-process **Spherical GPU** backend is fast; **EMSphInx** is a CPU path that
  runs inside **WSL** and is correspondingly slower. If GPU runs stall or throughput
  drops to near zero, a previous job may still hold VRAM — use **Release GPU**.
- **The *Embedding / FAISS* method is not implemented** in this build. The option
  appears in the dropdown but the backend rejects it with a clear error rather than
  silently running Hough.
- **Indexing a subset is position-safe.** Region and chemistry-mask selections are
  written back into the full-scan grid at their original coordinates, so partial
  results overlay correctly on the whole map.
- **Pseudo-symmetry.** For pseudo-symmetric cubic approximants the spherical
  correlation can land on the wrong symmetry variant; the result then takes its
  orientation from Hough where needed, and Pattern Match offers a manual
  variant-flip tool (see [Pattern Match](PatternMatch.md)).
- **Results live in memory.** Restarting the backend clears stored results; the app
  re-activates a saved result automatically when you reload the file it belongs to,
  but a fresh backend start requires re-indexing.
- **Spherical detector geometry.** When a file's pixel size produces a detector
  width outside EMSphInx's valid range, the backend auto-substitutes a pixel size
  (and logs it). Set a real pixel size in PC Refinement for physically accurate
  geometry.

## EDS-guided phase assignment (e.g. Al vs Si)

Some phases are almost identical crystallographically but clearly different
chemically — the classic case being **fcc Al and diamond-cubic Si**, both cubic
with the same symmetry. Their Kikuchi patterns look nearly the same, so
pattern-only indexing often labels Si particles as Al. When the loaded file has
**EDS**, you can let the local chemistry break the tie.

- Each selected phase gets an **"EDS influence" slider** (0–100 %). It only appears
  when EDS is available for the active file.
- **Default is 0 % for every phase**, which means *no change* — indexing is exactly
  as it was. Turn a phase up (e.g. set **Si** to 50–75 %) to have its EDS chemistry
  bias the phase choice **toward the chemically-consistent phase without overriding
  a clear pattern match**. It acts strongest exactly where the pattern is ambiguous.
- It works for **Spherical, Dictionary, and Hough** (a chemistry-active multi-CIF
  Hough run indexes each phase separately, so it is slower — only enabled when a
  strength is set).
- The expected composition comes from each phase's **formula** automatically; you
  do not need to enter anything.
- After the run, the log reports **`EDS chemistry prior: adjusted N pixels`** — how
  many pixels the chemistry moved off the pattern-only choice.
- **No EDS, or all sliders at 0 ⇒ bit-identical** to standard indexing.

Practical tip: leave well-behaved phases at 0 and only raise the discriminating
element (Si here). The chemistry is a **tie-breaker**, not an override — it
resolves ambiguous pixels but cannot invent a phase whose pattern is absent.
Because EDS is spatially coarser than EBSD, compositions at particle edges are
mixed; the soft weighting is designed to tolerate this.
