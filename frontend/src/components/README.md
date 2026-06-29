# `frontend/src/components` — Orienta UI Pages

This directory holds the **React UI** of Orienta, an EBSD (Electron
Backscatter Diffraction) pattern-analysis desktop application built on the
kikuchipy / orix / diffsims scientific stack. Each subdirectory is one
**feature page** (a screen reachable from the left sidebar) or a small set of
**shared building blocks** used across pages. Pages talk to the FastAPI backend
over HTTP/WebSocket through `../services/api.js` and read/write shared client
state via the Zustand stores in `../stores/`.

> Looking for **how to use** a feature rather than where its code lives? See the
> end-user how-to docs under [`../../../docs/user-guide/`](../../../docs/user-guide/).
> This file documents the **code layout** only.

## How this fits the architecture

Orienta is an **Electron shell → React frontend → FastAPI backend → scientific
layer** application. This directory is the React frontend's view layer:

- [`../App.jsx`](../App.jsx) is the application shell. It defines the sidebar
  page list (`SIDEBAR_PAGES`), lazy-loads each page below, and keeps every page
  mounted in a stacked layout (only the active one is visible). Keyboard
  shortcuts `Ctrl+1..0` switch pages; `Ctrl+H` toggles the HDF5 Viewer overlay.
- Pages are **presentational + orchestration only** — all numerical/scientific
  work happens in the backend. A page fetches data, renders it, and posts user
  actions back through `../services/api.js`.
- Cross-cutting client state lives in [`../stores/`](../stores/); reusable
  styled primitives (Button, GroupBox, splitter, etc.) live in
  [`../theme/`](../theme/); all user-facing strings are i18n keys
  (en / de / ja / zh) resolved via `react-i18next`.

## Feature pages (sidebar order)

| Folder | Sidebar entry | Purpose (one line) |
|--------|---------------|--------------------|
| [`Dashboard/`](./Dashboard/) | Dashboard | Landing page: workflow-ordered module cards and a recent-files list. |
| [`EBSDViewer/`](./EBSDViewer/) | EBSD Viewer | Load an EBSD scan, browse patterns, and apply pattern processing (background removal, contrast, CLAHE, detector mask). |
| [`EDS/`](./EDS/) | EDS Analysis | Composite + tiled view of EDS element maps with quantification, region averages, and chemistry-based phase suggestions. |
| [`PCRefinement/`](./PCRefinement/) | PC Refinement | Calibrate the pattern centre (global and pixel-wise) — the geometry indexing depends on. |
| [`CrystalDatabase/`](./CrystalDatabase/) | Crystal DB | Manage CIF files and convert them to EMsoft `.xtal` structures (incl. the Debye–Waller table). |
| [`Simulation/`](./Simulation/) | Simulation | Configure and launch master-pattern / SHT forward simulations (EMsoft via WSL). |
| [`DatabaseBrowser/`](./DatabaseBrowser/) | Database | Browse the simulated-file library (SHT, MC, master H5, CIF, XTAL) with preview and local-cache controls. |
| [`Indexing/`](./Indexing/) | Indexing | Run orientation indexing (Hough / Dictionary / Spherical) on a region or whole scan, with multi-phase comparison. |
| [`PhaseMap/`](./PhaseMap/) | Phase Maps | Layered phase / IPF / quality map viewer with per-layer opacity, blend, diagnostics, and refinement overlays. |
| [`Analysis/`](./Analysis/) | Analysis | MTEX-equivalent post-processing: grain reconstruction, size, deformation (KAM), texture, recrystallization, Excel export. |
| [`Batch/`](./Batch/) | Batch | Step-by-step wizard to queue indexing across many files (files → PC → preprocess → phases → method → run). |
| [`Refinement/`](./Refinement/) | Refinement | Inspect indexed pixels and apply manual orientation/phase overrides on the phase map. |
| [`CrystalHint/`](./CrystalHint/) | Crystal Hint | Detect crystal symmetry/lattice from patterns and suggest candidate phases (single-pixel, region, and quality-check modes). |
| [`MLHub/`](./MLHub/) | ML Hub | Train and run machine-learning models for pattern classification/prediction. |
| [`Settings/`](./Settings/) | Settings | System configuration: server mode, manual path overrides, install wizard, API keys. |
| [`HDF5Viewer/`](./HDF5Viewer/) | HDF5 Viewer (Ctrl+H) | Overlay tool to open any H5OINA/HDF5 file and inspect patterns, EDS, electron images, and metadata. |
| [`DictionaryGPU/`](./DictionaryGPU/) | Dictionary (GPU) | Stand-alone tool to generate a GPU dictionary from a master pattern (detector + orientation settings). |

## Components used inside pages (not direct sidebar routes)

| Folder | Purpose (one line) |
|--------|--------------------|
| [`PatternMatch/`](./PatternMatch/) | Linked experiment/simulation pattern view + publication figure-export composer, used by the Indexing phase-test dialog. |
| [`PoleFigure/`](./PoleFigure/) | Pole-figure renderer used inside Phase Maps and in a detached pole-figure window (`?view=polefigure`). |

## Shared building blocks

| Folder | Purpose (one line) |
|--------|--------------------|
| [`shared/`](./shared/) | App chrome shared across pages: `Sidebar`, `StatusBar`, `ToastContainer`, `DevPanel`. |
| [`common/`](./common/) | Small reusable widgets: `FileSwitcher`, `InfoTooltip`, `CoordinateSystemPanel` / `OrientationHelperSvg` (+ presets). |

## Notes for working in this directory

- **One page = one folder.** The folder's `*Page.jsx` (or, for the viewers,
  the like-named `EBSDViewer.jsx` / `HDF5Viewer.jsx` / `Dashboard.jsx`) is the
  entry component; everything else in the folder is a sub-component or hook for
  that page. Co-located `*.test.jsx` files are Vitest tests.
- **Adding a page:** create the folder + entry component, then register it in
  [`../App.jsx`](../App.jsx) (`SIDEBAR_PAGES` + the lazy import + a `<div data-page=…>`
  cell) and add its `nav:pages.<id>` label/tooltip to the i18n files.
- **No scientific logic here.** Keep heavy computation in the backend and call
  it through [`../services/api.js`](../services/api.js); use
  [`../stores/`](../stores/) for shared state and
  [`../theme/`](../theme/) for styled primitives. All visible text must go
  through `react-i18next` keys (en / de / ja / zh).
- **Run the frontend** from the `frontend/` root: `npm run dev` (dev server) or
  `npm run build`; tests with `npm run test`. The backend must be running for
  pages to load data (see the project root `README.md`).
