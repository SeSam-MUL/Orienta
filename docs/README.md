# Orienta — `docs/`

This directory holds the documentation for Orienta, an EBSD (electron backscatter
diffraction) pattern-analysis desktop application built on the kikuchipy / orix /
diffsims scientific stack with an Electron + React front end and a FastAPI back
end. It covers two audiences: **end users** (the per-module user guide) and
**developers/contributors** (the architecture overview and the on-disk
data-format reports). For day-to-day code orientation, the project root
`README.md` is a good first stop; this folder goes one level deeper.

## Start here

- **Using the app?** Read the **[User Guide](user-guide/README.md)** — one page per
  module, plus a typical end-to-end workflow.
- **Working on the code?** Read the **[Architecture Overview](ARCHITECTURE.md)** —
  the layer diagram, request flow, and what each root module does.

## Contents at a glance

| File / subdir | What it is |
|---|---|
| [`user-guide/`](user-guide/README.md) | End-user documentation, one page per application module (loading data, calibration, indexing, maps, analysis, crystal data, simulation, settings) with a typical workflow ordering. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Developer-oriented map of the whole system: the Electron → React → FastAPI → scientific-stack layers, how a typical action flows through them, and what each root-level Python module does. |
| [`h5oina_structure_report.md`](h5oina_structure_report.md) | Annotated map of the Oxford Instruments Aztec **`.h5oina`** HDF5 layout — EBSD data/header groups, pattern arrays, pattern-center fields, phase table, and the EDS spectra — with notes on what each field means for a kikuchipy import. |
| [`light_h5_format.md`](light_h5_format.md) | Full specification of the compact **light `.h5`** indexing export (`result_<stem>_light.h5`): the group/dataset tree, the `phase_id` convention, the single-phase vs. multi-phase layouts, the version history, and how to read it back in this app, in MATLAB/MTEX, or in Python/orix. |
| [`ipf-colour-maps.md`](ipf-colour-maps.md) | The complete IPF colour-map story: pseudo-symmetry variant handling (Hough substitution, map-wide render-verified unification, manual grain flip), the low-symmetry colour-key discontinuity and the grain-consistent v2 colouring, the per-phase IPF view — with measured evidence, literature grounding, and a what-to-use-when recipe. |

## Per-directory READMEs across the repo

Each major source directory has its own `README.md` describing what lives there
and how it is organised:

| Directory | What it covers |
|---|---|
| [`../backend/README.md`](../backend/README.md) | The FastAPI service and compute engines — REST + WebSocket API over the scientific stack and Orienta's GPU pipelines. |
| [`../frontend/README.md`](../frontend/README.md) | The React + Vite single-page application: the UI layer, pages, and zustand state stores. |
| [`../electron/README.md`](../electron/README.md) | The Electron desktop shell that spawns the backend and opens the app window. |
| [`../analysis/README.md`](../analysis/README.md) | MTEX-equivalent EBSD post-processing: grain reconstruction, grain size, deformation, texture, recrystallization. |
| [`../simulation/README.md`](../simulation/README.md) | The bridge to the EMsoft / EMSphInx toolchain (master-pattern simulation and file sync). |
| [`../tools/README.md`](../tools/README.md) | Standalone, UI-free data utilities (HDF5 patterns, indexing results, phase maps). |
| [`../Ai_Ml/README.md`](../Ai_Ml/README.md) | The EBSD-AI machine-learning phase-prediction component. |

## How this fits the overall architecture

Orienta's runtime is layered: an Electron shell hosts a React front end that talks
over HTTP/WebSocket to a FastAPI back end, which in turn drives the scientific
layer (kikuchipy, orix, diffsims). The two **format reports** in this directory
document the boundary data formats that cross those layers and the boundary to the
outside world:

- `h5oina_structure_report.md` describes the **input** side — the vendor files the
  back end ingests.
- `light_h5_format.md` describes a key **output** side — the portable result file
  the app exports for sharing and for MTEX/MATLAB interoperability. It links
  directly to the writer/reader code in `backend/api/`.

## Use notes specific to this directory

- These are **documentation files only** — nothing here is imported or executed by
  the application. Editing them has no effect on runtime behavior.
- Code references inside these docs use **relative links** back to the source tree
  (e.g. `../backend/api/...`); they resolve from this `docs/` directory.
- To understand a current behavior, prefer the format reports and the live source,
  which are authoritative when documentation and code disagree.
