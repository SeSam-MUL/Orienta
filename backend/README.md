# `backend/` — Orienta FastAPI service and compute engines

This directory holds the Python backend of **Orienta**, an EBSD (Electron
Backscatter Diffraction) pattern-analysis desktop application. The backend
exposes the scientific stack (kikuchipy / orix / diffsims plus Orienta's own
GPU pipelines) over a REST + WebSocket API that the React/Electron frontend
talks to. It is a thin web layer on top of the heavier numerical packages that
live both here (`spherical_gpu/`, `dict_gpu/`, `dictionary_gpu/`, `forward_sim/`)
and at the project root (`safe_loader.py`, `ebsd_utils.py`, `analysis/`,
`simulation/`, `tools/`).

## Top-level files and subpackages

| Path | Description |
|------|-------------|
| [`api/main.py`](api/main.py) | The FastAPI application. Creates the `app`, wires CORS, a request-timing middleware, the `/api/health` and `/api/shutdown` endpoints, the `/ws` WebSocket for live progress and dev-panel logs, the `lifespan` startup hooks (kikuchipy prewarm, orphaned-EMsoft reaper, optional frontend watchdog), and `include_router(...)` for every route module. In production it also serves the built React app from `frontend/dist`. |
| [`api/`](api/) | The web layer: `routes/` (one router per feature area — indexing, phase_map, eds, simulation, database, …), `services/` (stateful helpers: HDF5 session, result/phase-map stores, GPU runners, exporters, Crystal Hint, orientation-frame correction, …), `models/` (shared Pydantic models), `log_broadcast.py` (pushes log records to the dev panel over the WebSocket), and `data/`. |
| [`spherical_gpu/`](spherical_gpu/) | GPU-accelerated **spherical indexing** (PyTorch port of the EMSphInx IndexEBSD pipeline). Public entry point `backend.spherical_gpu.api.gpu_spherical_index_patterns`; `backend.py` is the production wrapper, `runtime.py` does GPU detection and memory-aware batch sizing, `pseudosym.py` handles pseudo-symmetry resolution, and `_math/` / `pipeline/` hold the SHT correlation math and detector/forward/refine stages. |
| [`dict_gpu/`](dict_gpu/) | GPU **dictionary indexing** (Path A). Public entry point `backend.dict_gpu.gpu_dictionary_index_patterns`; `runtime.py` does CUDA/VRAM detection, `pipeline/` runs the indexer, and `_pcadi/` is the vendored (MIT) PCA/quantize/kNN projection core. |
| [`dictionary_gpu/`](dictionary_gpu/) | GPU **dictionary generation** — a standalone, kikuchipy-equivalent forward projector (Lambert / detector projection) used to build the simulated dictionaries the indexers match against. Public entry point `backend.dictionary_gpu.generate_dictionary_gpu`. (Distinct from `dict_gpu/`, which *consumes* dictionaries; this one *produces* them.) |
| [`forward_sim/`](forward_sim/) | GPU-native EBSD **forward model** — computes a phase's dynamical master pattern from EMsoft crystal data + a Monte-Carlo energy/depth distribution, in parallel to EMsoft (which stays the validation oracle). Organised into `crystal/`, `mc/`, `dynamical/`, `io/`, and `validate/` (NCC / pattern-parity checks vs EMsoft). |
| [`workers/`](workers/) | Reserved package for background workers; currently a namespace placeholder (empty `__init__.py`). Long-running jobs today are managed in `api/services/` (e.g. batch managers) and the root-level `simulation/` package. |
| [`tests/`](tests/) | Backend-local test package marker. The bulk of the project's pytest suite lives in the repository-root [`tests/`](../tests) directory. |
| [`requirements.txt`](requirements.txt) | The minimal web-layer dependencies (FastAPI, uvicorn, websockets, pydantic, httpx, pytest). The scientific stack (kikuchipy, orix, diffsims, torch, h5py, …) is declared in the project-root `requirements.txt`. |

## How it fits the Orienta architecture

```
Electron shell  (electron/main.js)
      |
React frontend  (frontend/src/)
      |  HTTP + WebSocket
FastAPI backend (this directory — api/main.py + routers)
      |
Compute engines (spherical_gpu/, dict_gpu/, dictionary_gpu/, forward_sim/)
      |
Scientific layer (root: safe_loader.py, ebsd_utils.py, analysis/, simulation/)
      |
Libraries (kikuchipy, orix, diffsims, torch, h5py, numpy)
```

`api/main.py` is the single composition point: it imports each router from
`api/routes/` and mounts it under a `/api/...` prefix. Routes delegate stateful
work to `api/services/` and heavy numerical work to the GPU subpackages above
and to the root-level scientific modules. Real-time progress for long
operations is broadcast back to the frontend over the `/ws` WebSocket via the
`ConnectionManager` in `main.py`.

## In-memory session & result model

The backend is **stateful but non-persistent** — state lives in module-level
data structures for the duration of the process:

- One open HDF5 file at a time, held by the session singleton in
  [`api/services/h5_session.py`](api/services/h5_session.py) (with an LRU
  pattern cache).
- The active EBSD signal, the registry of loaded files, and per-file processed
  datasets live as module-level state in
  [`api/routes/ebsd_viewer.py`](api/routes/ebsd_viewer.py).
- Indexing results are kept in a bounded in-memory registry
  (`_result_registry`) in [`api/routes/indexing.py`](api/routes/indexing.py);
  phase maps, calibrations, and other artifacts are similarly held in their
  respective `services/` stores.

There is **no database and no on-disk persistence of results**. Consequently,
restarting the backend wipes all loaded data and indexing/analysis results —
they must be reloaded/recomputed.

## Run / use notes

- **No auto-reload.** The backend deliberately does *not* run with
  `--reload`. Every Python change in this directory requires a manual
  kill + restart of the server, and (because state is in-memory only) EBSD
  data and indexing results must be reloaded afterward.
- **Start it** (from the project root):
  ```bash
  python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
  ```
  or run the production launcher `python start_app.py` (starts the backend and
  opens the app). See the root `README.md` for all entry points.
- **Health check:** `GET http://localhost:8000/api/health`.
- **Production frontend serving:** if `frontend/dist` exists, `main.py` mounts
  it at `/` (SPA fallback, with no-cache headers on `index.html`); otherwise the
  frontend is served separately by the Vite dev server.
- **Optional behaviours via environment variables:** `KIKUCHIPY_WATCHDOG=1`
  enables the frontend watchdog and the `/api/shutdown` endpoint (used in
  Electron mode); it is off by default so long batches/simulations are not
  killed when no browser is attached. `KIKUCHIPY_WATCHDOG_GRACE_SEC` overrides
  the watchdog grace period.
- **GPU is optional.** The `spherical_gpu` / `dict_gpu` / `dictionary_gpu` /
  `forward_sim` engines detect CUDA at runtime and fall back to CPU (or fail
  loudly in GPU-only mode) when no compatible GPU is present.
