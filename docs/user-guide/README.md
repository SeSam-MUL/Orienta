# Orienta — User Guide

This is the end-user documentation for **Orienta**, an EBSD (Electron Backscatter
Diffraction) pattern-analysis desktop application built on the kikuchipy / orix /
diffsims scientific stack. Each page below documents one module of the
application: what it does, the controls it offers, and how it fits into the
overall workflow.

If you are new, start with the [Dashboard](Dashboard.md) (the home screen) and the
[EBSD Viewer](EBSDViewer.md) (where you load a scan), then follow the typical
workflow at the bottom of this page.

---

## Data & viewing

| Page | What it does |
|---|---|
| [Dashboard](Dashboard.md) | Home screen and navigation hub — clickable cards for every module plus a recent-files list. |
| [EBSD Viewer](EBSDViewer.md) | Load an EBSD scan, browse its Kikuchi patterns on an overview map, and apply pattern preprocessing (background removal, frame averaging, contrast). |
| [HDF5 Viewer](HDF5Viewer.md) | Low-level explorer for the contents of an H5OINA / HDF5 file ("H5OINA Cockpit") — groups, datasets, patterns, and EDS spectra. |
| [EDS](EDS.md) | Turn per-pixel EDS X-ray counts into element maps and composition (counts / wt% / at%) for chemistry-aware analysis. |

## Calibration

| Page | What it does |
|---|---|
| [PC Refinement](PCRefinement.md) | Calibrate the pattern centre (PC) and detector geometry — the key prerequisite for correct indexing. |
| [Crystal Hint](CrystalHint.md) | Estimate which crystal systems and lattice parameters could have produced a pattern, before running a full index. |

## Indexing

| Page | What it does |
|---|---|
| [Indexing](Indexing.md) | Assign a crystal phase and orientation to each pixel (Hough, Dictionary, or Spherical methods). |
| [Batch](Batch.md) | Index many EBSD files against many phases in one unattended run. |
| [Dictionary (GPU)](DictionaryGPU.md) | Standalone GPU tool that generates a simulated EBSD pattern dictionary from a master pattern. |
| [Pattern Match](PatternMatch.md) | Verify an indexing result pixel by pixel — measured vs. simulated pattern, NCC difference image, and match score. |

## Maps & analysis

| Page | What it does |
|---|---|
| [Phase Map](PhaseMap.md) | Visualise an indexing result as a spatial, layered map (phase, IPF orientation, quality, EDS, …). |
| [Phase Refinement](Refinement.md) | Interactive phase-map editor for manually overriding wrong per-pixel phase assignments after multi-phase indexing. |
| [Pole Figure](PoleFigure.md) | Plot the crystallographic texture of an indexing result (opened from the Phase Map page). |
| [Analysis](Analysis.md) | Post-processing of an indexed scan — grain reconstruction, grain size, deformation, texture, and recrystallization (MTEX-equivalent). |
| [ML Hub](MLHub.md) | Train and run a machine-learning phase classifier for EBSD patterns. |

## Crystal data & simulation

| Page | What it does |
|---|---|
| [Crystal Database](CrystalDatabase.md) | Manage the crystal structures Orienta uses for indexing, including CIF→.xtal conversion. |
| [Simulation](Simulation.md) | Generate the simulated reference data (master patterns, SHT, dictionaries) that indexing needs, via the EMsoft/EMSphInx toolchain. |
| [Database Browser](DatabaseBrowser.md) | Central file manager for your local crystal/simulation database (structures, master patterns, SHT files). |

## Settings

| Page | What it does |
|---|---|
| [Settings](Settings.md) | Configure system dependencies and preferences — system-status health check, install wizard, compute mode, and appearance. |
| [Reporting a problem](ReportingProblems.md) | Package version, logs and your description into one zip to attach to a bug report. |

---

## Typical workflow

A common end-to-end session moves through the modules in roughly this order:

1. **[EBSD Viewer](EBSDViewer.md)** — load your H5OINA/HDF5 scan and preprocess the patterns.
2. **[EDS](EDS.md)** *(optional)* — inspect per-pixel chemistry to inform phase selection.
3. **[Crystal Database](CrystalDatabase.md)** / **[Simulation](Simulation.md)** — make sure the candidate phases exist as structures and have simulated reference data; **[Crystal Hint](CrystalHint.md)** can suggest candidates.
4. **[PC Refinement](PCRefinement.md)** — calibrate the pattern centre / detector geometry.
5. **[Indexing](Indexing.md)** (or **[Batch](Batch.md)** for many files) — assign phase and orientation to each pixel.
6. **[Pattern Match](PatternMatch.md)** — verify the result pixel by pixel.
7. **[Phase Map](PhaseMap.md)** / **[Phase Refinement](Refinement.md)** — visualise and, if needed, correct the phase assignments.
8. **[Analysis](Analysis.md)** & **[Pole Figure](PoleFigure.md)** — quantify grains, deformation, texture, and recrystallization, and export results.

The **[Dashboard](Dashboard.md)** lets you jump to any of these at any time, and
**[Settings](Settings.md)** is where you first set up the optional WSL/EMsoft/GPU
dependencies used by simulation and spherical indexing.
