# Indexing

## What it does

The Indexing page assigns a **crystal phase and orientation** to each pixel of a
loaded EBSD scan by matching its Kikuchi pattern against candidate phases. It is
the heart of the workflow: its output (a per-pixel orientation/phase map, the
*CrystalMap*) feeds the [Phase Map](PhaseMap.md), [Pole Figure](PoleFigure.md),
and EBSD Analysis tools.

Three indexing methods are available:

- **Hough indexing** — the standard band-detection approach. Needs a **CIF** file
  per phase. Fast, good for cubic/well-resolved patterns. The Radon-transform band
  detection and the band indexing are done by
  [PyEBSDIndex](https://github.com/USNavalResearchLaboratory/PyEBSDIndex)
  (Rowenhorst, Callahan & Ånes, *J. Appl. Cryst.* **57**, 3–19, 2024,
  [doi:10.1107/S1600576723010221](https://doi.org/10.1107/S1600576723010221));
  cite it when you report Hough-indexed results. The *Citations for this result*
  panel on the Phase Map page lists it automatically for a Hough run.
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
   one with its × button. A phase file that is not in the library can be added
   with the path field at the bottom of the picker: paste or type its full path
   and press **Add** (the desktop app also has a **Browse…** button). The app
   checks that the file exists, has the extension the method needs and can be
   read as a phase, and says what is wrong if not. This is the only way to add
   a phase of your own when the app runs in a normal browser (`start_app.py`),
   which has no native file dialog.
4. If the app flags **degenerate** phases (entries EBSD cannot tell apart), use
   **Reduce phases** to trim the list — fewer near-identical candidates means a
   cleaner result.

### 3. Set method parameters

5. **Hough:** set *Bands* (how many of the strongest Radon peaks of each pattern
   are used), *t-sigma* and *r-sigma* (the widths, in Radon bins, of the band
   detector along theta and across the band). Hover a label for what it does and
   its default; Orienta's defaults (12, 2, 2) differ from PyEBSDIndex's own
   (9, 1, 1.2). Each phase card also has a **Reflector families** table (see
   below).
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
- **Pseudo-symmetry.** Cubic approximants (point group m-3, e.g. alpha-AlFeMnSi) and
  orthorhombic phases (mmm, e.g. the S phase) are indexed by the spherical method
  directly — measured against EMSphInx and on real S-phase patterns (September 2026).
  Only for point group -43m (e.g. Mg17Al12), whose patterns look almost the same after
  a 90° turn, does the result take its orientation from Hough. Pattern Match offers
  a manual variant-flip tool for any phase
  (see [Pattern Match](PatternMatch.md)).
- **Results live in memory.** Restarting the backend clears stored results; the app
  re-activates a saved result automatically when you reload the file it belongs to,
  but a fresh backend start requires re-indexing.
- **Spherical detector geometry.** When a file's pixel size produces a detector
  width outside EMSphInx's valid range, the backend auto-substitutes a pixel size
  (and logs it). Set a real pixel size in PC Refinement for physically accurate
  geometry.

## Hough: reflectors, memory, and why the app can refuse a phase

Hough does not compare full patterns. It detects a handful of bands per pattern
(`n_bands`, 12 by default) and identifies the orientation from the **angles
between them**, using a lookup table built once per run: the **band-triplet
library**. PyEBSDIndex sizes that library from the number of **reflector
families** the phase contributes, and it grows roughly with the **cube** of the
number of distinct inter-pole angles.

That is why the same number of families costs wildly different amounts.
Measured on a 156x128 detector:

| Phase (CIF) | Symmetry | Families | Library |
|---|---|---:|---:|
| Al (Fm-3m) | m-3m | 64 | a few MiB |
| MgCuAl2 (Cmcm) | mmm | 70 | 0.59 GiB |
| beta-AlFeSi | 2/m | 70 | 3.55 GiB |
| Al7FeCu2 **stored as P 1** | 1 | 70 | **44.7 GiB** |

Symmetry is what makes a phase cheap: in a cubic phase one family stands for up
to 48 equivalent poles, so a few families describe many directions. With no
symmetry each family is a **single** pole, and the angle count — and with it the
library — explodes.

**On Windows the memory is claimed the moment it is requested**, whether or not
it is ever written. So an impossible request is not a slow run: it is a machine
that starts swapping, and unrelated things begin to fail. Orienta therefore
**refuses** a library that does not fit rather than attempting it. The budget is
half of what the machine actually has free (never a fixed number, so it adapts
to the computer you are on), and the refusal names the phase, its symmetry, and
what each reflector count would cost.

### Reflector families: which planes a phase is indexed with

Hough indexing matches the detected bands against the angles between a short
list of plane **families**. The default list is built by a rule (spacing
*d* ≥ 1 Å, |F| above a tenth of the strongest, at most 70 rows) and PyEBSDIndex
then drops any family whose pole is a multiple of an earlier one: for aluminium
the six families {111} {200} {220} {311} {222} {400} become four, because {222}
repeats {111} and {400} repeats {200}.

Each phase card of a Hough run (and each loaded phase on the **PC Refinement**
page) has a **Reflector families** card. Open it to see every candidate family
with its spacing, relative |F| and multiplicity, and which ones are used.

- **Untick** a family to leave it out, or tick one back in. Unticking {111}
  while {222} is still ticked changes nothing, because {222} is the same pole
  and takes its place; the table shows that.
- **Strongest N** keeps the N families with the largest structure factor.
- **Add family** takes `hkl` (or `hkil` for hexagonal and trigonal phases) and
  refuses a reflection the crystal forbids.
- **Rule for the default list** changes *min d* and the |F| threshold.
- **Reset to default** removes your selection.

A phase without a selection uses the default list. Hough indexing needs
at least two distinct families, and PyEBSDIndex cannot build its library from
every pair; such a selection is refused when you make it, with the reason. The
choice is stored per phase on the backend, so it applies to every Hough build of
that phase, on both pages and in batch runs, until you reset it. A selection made
for a different crystal (another space group or cell) is never applied; the card
says so and offers the reset. The run log names a phase that used its own
selection and lists the families it used.

Measured on SampleB (Al, 600 patterns, 20x30 region): the default list written
out as a selection gives the **identical** result; unticking {111} and {222}
changes the CI of every pixel, leaves one pixel unindexed, and moves the
orientation of 52 of the other 599 by more than 5 degrees. Compare the map against the default before relying on a
reduced list, exactly as for the row count below.

### The reflector rows control

Each phase card in a **Hough** run carries a **Reflector rows** dropdown. It
counts **rows** of the reflector list (every symmetry equivalent and both signs
of a reflector is one row), not families: aluminium's 64 rows are six families,
and the first 24 rows leave out {111}. It is a memory guard for runs without a
family selection, and is not used for a phase that has one. It lists
every option with its price, e.g. `all 70 — 44.73 GiB (too big)`, `40 — 1.45
GiB`, `32 — 0.32 GiB`. Options that do not fit stay selectable and are marked;
hiding them would read as "this was never possible" and the run would then fail
with no visible reason.

**Lowering it is not free, and Orienta will not do it for you.** Measured on Ni
(m-3m, 58 families, 800 real patterns):

| Families | Median fit | Indexed | Deviation from the full set |
|---:|---:|---:|---:|
| 58 (all) | 0.793 deg | 495/800 | reference |
| 40 | 0.793 deg | 495/800 | **0.000 deg** |
| 32 | 0.793 deg | 495/800 | **0.000 deg** |
| 24 | 0.756 deg | 192/800 | **119.7 deg** |
| 16 | 180 deg | 4/800 | broken |

Down to a point the result is *identical*; past it the run collapses — and it
does so **quietly**, returning orientations that still look like data. No
program can tell a safe trim from an unsafe one without indexing and comparing,
which is why the number is yours to set. **After lowering it, compare the map
against a run with more families before you rely on it.**

### If a phase is refused

1. **Check the symmetry first.** If the card says the CIF stores no symmetry
   (`P 1`), that is the real problem and it is fixable at the source: re-export
   the phase in its actual space group. Al7Cu2Fe as `P 1` needs 44.7 GiB; the
   same phase with its real tetragonal symmetry needs a fraction of that, and
   the reflector count never has to be touched.
2. **Lower the reflector count** for that phase, then verify as above.
3. **Free memory** — the budget follows what is available, so closing other
   programs raises it.
4. **Use Dictionary or Spherical indexing** for that phase, which build no
   band-triplet library at all.

An advanced override exists for the budget itself:
`ORIENTA_HOUGH_LIBRARY_BUDGET_MB` (megabytes). It raises or lowers the ceiling
for people who know their machine — it does not make an impossible allocation
possible.

## EDS-guided phase assignment (e.g. Al vs Si)

Some phases are almost identical crystallographically but clearly different
chemically, the classic case being **fcc Al and diamond-cubic Si**, both cubic
with the same symmetry. Their Kikuchi patterns look nearly the same, so
pattern-only indexing often labels Si particles as Al. When the loaded file has
**EDS**, you can let the local chemistry decide.

- One switch, **Use EDS chemistry for phase assignment**, turns it on for
  **every selected phase at once**. It appears only when the active file has EDS.
- **Off is the default**, and an off run is bit-identical to indexing without
  the feature: the field is not sent at all.
- The expected composition comes from each phase's **formula** automatically.
  You do not have to enter anything.
- It works with **Spherical, Dictionary and Hough**. A chemistry-active
  multi-phase Hough run indexes each phase separately, so it is slower. The
  **EMSphInx** backend ignores the chemistry; that run stays pattern-only and
  the page says so before you start.
- After the run the log reports **`EDS chemistry prior: adjusted N pixels`**,
  how many pixels the chemistry moved off the pattern-only choice.

This is not a gentle tie-breaker. A phase the chemistry rules out keeps a
weight of 0.05, a twentyfold penalty, so at ambiguous pixels the chemistry
effectively decides. What it cannot do is invent a phase whose pattern is
absent. **[The EDS chemistry prior](EDS-prior.md)** describes the weighting,
what "expected composition" means, the orientation rescue for particles the
pattern cannot see, and when to leave the switch off.

### There is no per-phase strength, on purpose

Until August 2026 each phase had its own 0 to 100 % slider. The weight is
`w = (1 - s) + s * chemistry_fit`, so a phase left at 0 keeps weight exactly
1.0 and is immune, while every phase you raise is penalised. Raising it on Al
and Si alone was measured to inflate an unrelated third phase from 1.5 % to
10 % of the map. Only two settings mean anything, off for everything and on
for everything, and that is what the page offers. Older guidance telling you
to raise one phase to 50 to 75 % predates that measurement.

### The pre-flight check

With the switch on, the page checks the dataset before it lets you start: that
EDS is present, that the EDS and EBSD grids line up, that the signal is strong
enough, and that the defining elements of the selected phases were measured at
all. A failing check **blocks Start** until you fix it or switch the chemistry
off.

The same panel lists each phase with its defining elements and a **possible
area**: an upper bound on how much of the map the chemistry cannot rule that
phase out of. It is a bound, not a predicted area fraction, and for a
single-element phase it is trivially weak, because traces of that element sit
everywhere.

Because EDS is spatially coarser than EBSD, compositions at particle edges are
mixed. The soft weighting is designed to tolerate that.
