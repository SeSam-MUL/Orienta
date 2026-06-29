# Phase Refinement

## What it does

Phase Refinement is an **interactive phase-map editor**. After a multi-phase
indexing run, some pixels are assigned the wrong phase — most often where the
confidence (CI) between competing phases is close. This page lets you inspect any
pixel, see how each candidate phase scored, and **manually override** the phase
assignment for individual pixels.

For the loaded result it shows:

- The reconstructed **phase map** (colour-coded by phase), with an optional
  **uncertainty overlay** that dims low-confidence pixels.
- A **pixel inspector** that lists, for the clicked pixel, every candidate phase
  ranked by its confidence index (CI), with a visual CI bar.
- Summary **statistics** for the result (total pixels, number of overrides,
  average CI, count of low-CI pixels).

Manual corrections are written back into the result's checkpoint so the corrected
phase map can be used downstream.

## When to use it

Use Phase Refinement after a **multi-phase indexing/batch run**, when you need to
clean up specific mislabelled pixels by hand:

- A small grain or particle was assigned to the wrong phase and you want to
  reassign it to the runner-up candidate.
- You are reviewing a phase map for publication and want to fix obvious,
  isolated misindexed pixels.

It fits after [Indexing](../README.md) (specifically the multi-phase / batch
phase-refinement workflow) and before final phase-map export and analysis. It
operates on a saved multi-phase checkpoint rather than the live in-memory result.

## How to use it — step by step

1. **Load a result.** In the top bar, enter the **file stem** of an indexed
   dataset (the base name of its `…_multiphase.h5` checkpoint) and, if needed, a
   **search directory** where the checkpoint lives. Click **Load**.
   - When you arrive here from the batch dashboard's "Open in Refinement" action,
     the file stem and directory are filled in and loaded automatically.
2. The **phase map** appears on the left with a colour **legend** of phase names.
   Toggle **Uncertainty overlay** (top right) to dim pixels below the current CI
   threshold so the doubtful regions stand out.
3. **Click a pixel** on the map. The right-hand **Inspector** lists that pixel's
   candidate phases sorted by CI, each with a CI value and bar. The top candidate
   is highlighted.
4. To correct the pixel, click **Assign** next to the phase you want. The
   override is applied immediately, the map and statistics refresh, and the
   override count increases.
5. Use the bottom toolbar to:
   - Adjust the **CI <** threshold slider — this controls which pixels the
     uncertainty overlay treats as "low confidence".
   - See the running **override count**.
   - **Export** the result (opens the result's export URL when available).
6. **Clear** (top bar) unloads the current result so you can load another.

## Inputs & outputs

**Inputs**

- A **multi-phase indexing checkpoint** (`<file_stem>_multiphase.h5`) produced by
  the multi-phase / batch phase-refinement workflow, located either in the
  search directory you provide or in the app's default result locations.
- Your manual phase choices (per pixel).

**Outputs**

- A **manual-override layer** written back into the checkpoint (per-pixel phase
  id plus a source tag recording that the override was made by clicking). This
  persists with the checkpoint.
- The refreshed on-screen phase map, legend, and summary statistics.
- An exported result, when you use the Export button (uses the export URL the
  backend provides for the loaded result).

## Tips & notes

- **Only per-pixel ("Pixel") override is available in this build.** The
  underlying design also envisaged *Auto*, *Region*, and *Threshold* batch-edit
  modes, but those are **not wired** and are hidden in the shipped UI. Corrections
  are made one pixel at a time by clicking and assigning.
- **There is no separate "Save".** Each assignment is persisted to the checkpoint
  immediately, so there is no save button — overrides are not lost if you move
  away from the page.
- **It works on a saved checkpoint, not the live result.** You must point it at
  the dataset's `…_multiphase.h5`; if none is found, loading fails with a "no
  multiphase result found" message. If the checkpoint is missing its grid shape,
  overrides cannot be applied.
- **The CI slider only affects the overlay**, i.e. which pixels are visually
  dimmed — it does not change any phase assignments by itself.
- **Distinct from joint orientation + PC refinement.** This page edits *phase
  labels*. The separate joint orientation + pattern-centre refinement (Phase B)
  is a different, advanced feature reached from the Phase Map page and is not part
  of this editor.
