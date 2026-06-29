# Orienta — Architecture Overview

Orienta is a desktop application for analysing EBSD (Electron Backscatter
Diffraction) patterns. It indexes Kikuchi patterns, identifies crystal phases,
refines the pattern centre, and produces phase / orientation maps. It is built
on the [kikuchipy](https://kikuchipy.org/), [orix](https://orix.readthedocs.io/)
and [diffsims](https://diffsims.readthedocs.io/) scientific Python stack.

This document is a developer-oriented map of the whole system: the layers, how a
typical action flows through them, what each root-level Python module does, and a
few cross-cutting facts that are easy to get wrong.

---

## 1. Layer diagram

Orienta is a thin desktop shell wrapping a local web app: a React single-page
frontend talks over HTTP + WebSocket to a FastAPI backend, which in turn drives
the scientific Python libraries.

```
+----------------------------------------------------------------+
|  Electron shell                  electron/main.js, preload.js  |
|  - spawns the Python backend as a child process                |
|  - opens a BrowserWindow on the backend / dev-server URL       |
|  - native file/folder/save dialogs via IPC                     |
|  - renderer-crash recovery (keeps long batches alive)          |
+----------------------------------------------------------------+
                              | loads URL
                              v
+----------------------------------------------------------------+
|  React frontend (Vite SPA)       frontend/src/                 |
|  - components/   ~20 feature pages (EBSDViewer, Indexing,      |
|                  PhaseMap, EDS, PatternMatch, Simulation, ...)  |
|  - services/api.js   typed wrappers over every REST endpoint    |
|  - stores/       Zustand stores (data, results, progress, ...)  |
+----------------------------------------------------------------+
                |  HTTP (REST)            ^  WebSocket /ws
                v                         |  (progress, dev logs)
+----------------------------------------------------------------+
|  FastAPI backend                 backend/api/                  |
|  - main.py        app, CORS, lifespan, static SPA mount, /ws    |
|  - routes/        ~23 routers (one per feature area)            |
|  - services/      h5 sessions, image utils, refinement, ...     |
|  - spherical_gpu/, dict_gpu/   GPU indexing pipelines           |
+----------------------------------------------------------------+
                              | imports
                              v
+----------------------------------------------------------------+
|  Scientific layer                root-level *.py + libraries   |
|  - root modules   loaders, controllers, EDS, phase metadata     |
|  - kikuchipy / orix / diffsims / hyperspy / numpy / scipy       |
|  - PyEBSDIndex (Hough), scikit-learn, h5py                      |
|  - optional: EMsoft/EMSphInx via WSL (master patterns, SHT)     |
+----------------------------------------------------------------+
```

### Entry points

| Mode | Command | What it does |
|------|---------|--------------|
| Desktop app | `python start_desktop.py` | Runs `npm run electron:dev` (Vite + Electron + backend together). |
| Browser (production) | `python start_app.py` | Starts the backend, which serves the pre-built React SPA, then opens a browser. |
| Browser (dev) | `python start_app.py --dev` | Starts the backend plus the Vite dev server (`http://127.0.0.1:5173`). |
| Backend only | `python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000` | Backend on port 8000; API docs at `/docs`. |

In packaged / production mode the React build (`frontend/dist`) is mounted as a
static SPA directly by FastAPI (see the static mount at the end of
[`backend/api/main.py`](../backend/api/main.py)), so the whole app is reachable
from a single origin on port 8000. The Electron window simply loads that URL.

---

## 2. Request / data flow for a typical session

A representative workflow — **load a file → view patterns → index → phase map** —
touches every layer:

1. **Load file.**
   The user picks an `.h5oina` / `.h5` file (native dialog via Electron IPC, or a
   typed path). The frontend POSTs to `/api/ebsd/load`. The backend loads the
   scan through [`safe_loader.py`](../safe_loader.py) (kikuchipy fast path, with a
   robust [`unified_loader.py`](../unified_loader.py) fallback adapted via
   [`ebsd_adapter.py`](../ebsd_adapter.py)) and keeps the resulting kikuchipy
   `EBSD` signal in an **in-memory registry** inside the `ebsd_viewer` route
   module. Large files load lazily and report staged progress, which the frontend
   polls and shows in a progress modal.

2. **View patterns.**
   The EBSD Viewer page requests rendered overviews and single-pattern PNGs
   (`/api/ebsd/...`): mean / band-contrast maps, per-pixel patterns, optional
   background removal, autocontrast, CLAHE and a detector mask. These operate on
   the cached signal — no re-read from disk.

3. **(Optional) refine the pattern centre.**
   The PC Refinement page uses [`pc_controller.py`](../pc_controller.py) and
   [`ebsd_utils.py`](../ebsd_utils.py) (`optimize_pc`, `create_indexer`) to fit
   the PC on selected good patterns; the refined PC is written back so subsequent
   indexing uses it.

4. **Index.**
   The Indexing page selects which pixels to index (whole image, a region, or an
   EDS-chemistry mask) and a method (Hough, Dictionary, or Spherical), then POSTs
   to `/api/indexing/...`. [`indexing_controller.py`](../indexing_controller.py)
   runs the chosen engine: Hough/Dictionary via kikuchipy (CPU) or the GPU
   pipelines in `backend/dict_gpu/` and `backend/spherical_gpu/`; Spherical can
   also call EMSphInx through WSL. Progress streams over the `/ws` WebSocket.
   Results — including correct **back-mapping of partial selections to original
   grid positions** — are stored in an in-memory result registry keyed by a
   result id.

5. **Phase / orientation map.**
   The Phase Map page reads a stored result and renders layered maps
   (phase, IPF, band contrast, confidence, KAM, EDS overlays, …). Layer
   compositing is done client-side on a canvas; the backend supplies individual
   transparent RGBA layers. Results can be exported, or carried into the Analysis,
   Pattern Match, and Pole Figure pages — or saved as a `.kgproj` project bundle
   via [`project_manager.py`](../project_manager.py).

Throughout, real-time progress and developer log lines are pushed from the
backend to the frontend over the single `/ws` WebSocket; ordinary data requests
are plain REST calls described in `frontend/src/services/api.js`.

---

## 3. Root-level core modules

These live at the project root and form the scientific / controller layer that the
FastAPI routes import. One line each, with where it is used.

| Module | One-line role | Used by |
|--------|---------------|---------|
| [`ebsd_utils.py`](../ebsd_utils.py) | EBSD processing helpers: reflector preparation, Hough indexer creation, PC optimisation, CIF sanitisation. | `pc_controller`, indexing routes, GPU pipelines, several services. |
| [`ebsd_adapter.py`](../ebsd_adapter.py) | Converts `unified_loader` output into a kikuchipy `EBSD` signal (incl. vendor PC-convention handling). | `safe_loader`, EBSD load path. |
| [`unified_loader.py`](../unified_loader.py) | Vendor-agnostic HDF5 loader for EDAX `.h5` and Oxford `.h5oina` (patterns + detector/navigation metadata as dataclasses). | `safe_loader` fallback path. |
| [`safe_loader.py`](../safe_loader.py) | Hybrid loader: kikuchipy fast path with a robust `unified_loader` fallback, plus an Aztec H5OINA compatibility workaround. | EBSD load route; prewarmed at backend startup. |
| [`pc_controller.py`](../pc_controller.py) | Controller for phase/detector setup, pattern caching, and global pattern-centre refinement. | PC Refinement route. |
| [`indexing_controller.py`](../indexing_controller.py) | Orchestrates Dictionary / Hough / Spherical indexing, pixel selection, and back-mapping of partial results onto the full grid. | Indexing & batch routes. |
| [`ml_controller.py`](../ml_controller.py) | Bridge to the optional `ebsd_ai` ML subsystem (training store, model load/save, predictions); degrades gracefully if absent. | ML Hub route. |
| [`eds_utils.py`](../eds_utils.py) | Extracts EDS counts and converts counts → wt.% → at.% (simplified Cliff-Lorimer) for phase pre-selection. | EDS route, EDS overlay, pixel-chemistry services. |
| [`eds_overlay.py`](../eds_overlay.py) | Composites multiple EDS element maps into a single RGBA overlay (pure logic, no GUI deps). | EDS route. |
| [`phase_metadata.py`](../phase_metadata.py) | Extracts formula / space group / Pearson symbol / lattice from CIF, SHT, H5, XTAL files for phase-selection display and degeneracy checks. | Indexing & database routes, phase services. |
| [`path_utils.py`](../path_utils.py) | Cross-platform path handling (Windows ↔ WSL `/mnt/` conversion), filename sanitisation, database-folder discovery. | Simulation, database, file-sync paths. |
| [`project_manager.py`](../project_manager.py) | Save/load an Orienta session as a `.kgproj` bundle (phase maps, CrystalMaps, scores, masks + metadata). | Phase Map gallery / project save-load. |

---

## 4. Cross-cutting notes

A few facts that affect day-to-day development:

- **No backend auto-reload.** The backend is started with plain `uvicorn` (no
  `--reload`). Any change to Python code requires killing and restarting the
  backend process. After a restart, in-memory state — loaded EBSD data and
  indexing results — is gone and must be reloaded / re-run.

- **In-memory result registry (no persistence).** Loaded EBSD signals and
  indexing results are held in module-level dictionaries in the route modules
  (e.g. `_raw_signals` / `_loaded_files` / `_registry_by_file` in `ebsd_viewer`,
  and `_result_registry` in `indexing`). They are bounded (oldest entries evicted)
  and are **not** written to disk. A process restart clears everything; durable
  output goes through explicit export or the `.kgproj` project bundle.

- **Conda `ebsd` environment required.** The backend imports the heavy scientific
  stack (kikuchipy, orix, diffsims, hyperspy, PyEBSDIndex, scikit-learn, h5py).
  It must run under a Python environment that has these installed — the project's
  conda environment is named **`ebsd`**. The Electron launcher auto-detects a
  conda env that contains `kikuchipy` (+ `pyebsdindex`); you can override the
  interpreter with the `PYTHON_PATH` env var or a `.python_path` file in the
  project root.

- **Single origin in production.** When `frontend/dist` exists, FastAPI serves the
  SPA itself, so frontend and API share `http://127.0.0.1:8000` (no CORS issues).
  In dev mode the Vite server runs on `5173` and CORS is configured accordingly in
  [`backend/api/main.py`](../backend/api/main.py).

- **WebSocket-driven UI + watchdog.** Progress and dev logs are broadcast over
  `/ws`. In Electron mode a frontend watchdog can shut the backend down when no
  client is connected; `--headless` (or `KIKUCHIPY_WATCHDOG=0`) disables it so
  multi-hour batches and simulations survive a browser/renderer crash. The Electron
  shell also reloads a crashed renderer rather than killing the backend with it.

- **Optional WSL / EMsoft integration.** Spherical indexing via EMSphInx and
  master-pattern / SHT simulation via EMsoft run inside WSL. `path_utils.py`
  bridges Windows and WSL paths, and the backend reaps orphaned EMsoft processes
  on startup. These features are optional; core indexing (Hough, Dictionary,
  Spherical-GPU) works without them.
