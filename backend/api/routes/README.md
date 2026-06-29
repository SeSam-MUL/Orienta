# `backend/api/routes` — HTTP route handlers

This directory holds the FastAPI **route modules** that make up Orienta's REST/WebSocket API. Each file defines a `router = APIRouter()` whose path operations expose one functional area of the application (loading EBSD data, indexing, phase maps, simulation, the crystal database, and so on). The routers are mounted under stable prefixes (e.g. `/api/indexing`, `/api/ebsd`) by [`../main.py`](../main.py). Route handlers are deliberately thin: they validate the request, call into the services and scientific layers, and serialise the result — the heavy lifting lives in [`../services`](../services) and the project's scientific modules (`kikuchipy`, `orix`, `diffsims`, plus the in-repo `backend/spherical_gpu`, `backend/dict_gpu`, `analysis/`, `simulation/` packages).

## Route files

Each row lists the mount prefix it is given in [`../main.py`](../main.py) and a one-line summary of the endpoints it exposes.

| File | Mount prefix | Endpoints (summary) |
|------|--------------|---------------------|
| [`ebsd_viewer.py`](ebsd_viewer.py) | `/api/ebsd` | Load/switch/manage EBSD files, read patterns and overviews, datasets and metadata, deep-copy, and preprocessing (frame-average, background removal, autocontrast, CLAHE, signal mask) with progress polling. |
| [`h5_viewer.py`](h5_viewer.py) | `/api/h5` | Generic HDF5 explorer: open/close a file, browse the tree and attributes, read patterns, EDS maps/spectra, electron images, scan quality/defect/scalar maps, and a minimap. |
| [`virtual_images.py`](virtual_images.py) | `/api/ebsd` | EBSD-derived images for the EDS page: Virtual BSE (ROI pattern-intensity sum) and Band Contrast (native field or kikuchipy quality), returned as PNG. |
| [`indexing.py`](indexing.py) | `/api/indexing` | Core indexing engine: start/stop/status of Hough/Dictionary/Spherical runs (CPU & GPU), refinement, pattern-match preview, pseudo-symmetry variants & grain-flip, single-pixel phase test, forward-NCC, result management, import/export, and batch indexing. |
| [`pcrefinement.py`](pcrefinement.py) | `/api/pc` | Pattern-Centre refinement workbench: add/remove calibration patterns, set detector/phase, index single/all patterns, optimize and grid-calibrate the PC, drift modelling, and render previews. |
| [`refinement.py`](refinement.py) | `/api/refinement` | Two endpoint groups on one router: legacy multiphase phase-refinement on `_multiphase.h5` checkpoints (`/{file_stem}/...`), and joint orientation + pattern-centre refinement (`/compute`, `/resmooth`, `/cancel`, `/progress`, `/summary`). |
| [`phase_map.py`](phase_map.py) | `/api/phasemap` | Layered phase-map rendering (render, per-layer overlays, IPF key, available maps/directions), cleanup, export, phase legend/stats/adjacency, and probe/region-stats/linescan tools. |
| [`analysis.py`](analysis.py) | `/api/analysis` | Post-indexing (MTEX-equivalent) analysis: load a CrystalMap, quality stats/filtering, grain reconstruction, KAM/GOS/IPF map rendering, grain-size/deformation/texture/BC-GMM analysis, Aztec comparison, PC-drift, and Excel/batch export. |
| [`eds.py`](eds.py) | `/api/eds` | EDS chemistry: element maps, pixel/region quantification (Counts→Wt%→At%), CIF-phase suggestion, chemistry masks, auto-classify, phase-map painting (polygon/region) with indexing config, and probe/linescan tools. |
| [`crystal_hint.py`](crystal_hint.py) | `/api/crystal-hint` | Crystal-structure hinting from raw patterns: per-pixel and per-region symmetry/lattice analysis, quality check, Bravais presets, external (COD) search, CIF download, and library listing. |
| [`simulation.py`](simulation.py) | `/api/simulation` | EMsoft/forward-sim job control: config, system status & capabilities, start/stop/status/log of master-pattern/SHT simulations (CPU & GPU), history, scan-missing, batch jobs, server config, NML templates, and crystal picker. |
| [`database.py`](database.py) | `/api/database` | Crystal-structure database: browse, add/convert CIF↔XTAL, CIF/XTAL/SHT info & previews, sphere render, DWF table edit, build index, entries CRUD, delete, and server sync/resolve. |
| [`batch_v2.py`](batch_v2.py) | `/api/batch-v2` | Multi-phase batch indexing v2: scan folders, quick-load files (metadata only), PC status/copy, and batch create/start/pause/resume/stop/status/jobs/report. |
| [`pole_figure.py`](pole_figure.py) | (self, `/api`) | Pole-figure generation from the active CrystalMap (`GET /api/pole-figure`). |
| [`reference_frame.py`](reference_frame.py) | (self, `/api`) | Coordinate-system / reference-frame editing and a global state-version counter (`/api/frame`, `/api/state-version`). |
| [`calibration.py`](calibration.py) | `/api/calibration` | Per-dataset PC calibration store: list, get, set PC, delete, and propagate to the parent dataset. |
| [`dictionary_gpu.py`](dictionary_gpu.py) | `/api/dictionary-gpu` | Stand-alone GPU dictionary generation tool: generate, progress, list, and delete dictionaries. |
| [`forward_diagnostics.py`](forward_diagnostics.py) | `/api/forward-diagnostics` | On-demand SHT forward-diagnostic layers: compute (with progress/cancel), summary, anomaly browser, and thumbnails. |
| [`ml_hub.py`](ml_hub.py) | `/api/ml` | ML hub: model status/listing, prediction, training (with task polling), and clearing the model store. |
| [`settings.py`](settings.py) | `/api/settings` | Application settings backed by per-machine user config: server mode, manual binary paths, API keys (with test), and server dir creation. |
| [`install.py`](install.py) | `/api/install` | Install wizard: WSL status/install/user/password management and EMsoft installation streamed over a WebSocket. |
| [`system.py`](system.py) | `/api/system` | System probes — GPU detection and VRAM (`GET /api/system/gpu`). |
| [`__init__.py`](__init__.py) | — | Empty package marker. |

> Note: `ebsd_viewer.py` and `virtual_images.py` are both mounted under `/api/ebsd`; their paths do not collide.

## How this fits the Orienta architecture

```
Electron + React frontend
        │  HTTP / WebSocket
backend/api/main.py            ← mounts every router below under its prefix
        │
backend/api/routes/  (this dir)  ← request validation + serialisation only
        │
backend/api/services/          ← session/state, image utils, batch & refinement orchestration
        │
Scientific layer               ← kikuchipy / orix / diffsims, backend/spherical_gpu,
                                  backend/dict_gpu, analysis/, simulation/
```

The route layer is the public surface of the backend. Every endpoint the frontend calls is declared here (the frontend's matching JS wrappers live in `frontend/src/services/api.js`). Keep handlers thin and push reusable logic down into [`../services`](../services) so it can be shared and unit-tested independently of HTTP.

## Run / use notes

- Routers are not auto-discovered — adding a new module means importing it and calling `app.include_router(...)` with a prefix and tag in [`../main.py`](../main.py).
- The backend has **no auto-reload**: any change in this directory requires a backend restart to take effect.
- Once the backend is running, the live, always-current endpoint reference is the auto-generated OpenAPI UI at `http://localhost:8000/docs` (the `tags` set in `main.py` group the routes there). Health check: `GET /health`.
- Several routers depend on in-memory session state (the loaded EBSD file, the active indexing result, calibration). A restart clears that state, so data and results must be reloaded.
