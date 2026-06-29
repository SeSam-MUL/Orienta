# `backend/api/services` — Service-layer helpers

This directory holds the **service layer** of Orienta's FastAPI backend: the
reusable, framework-agnostic helpers that sit *between* the HTTP route handlers
in [`../routes`](../routes) and the scientific computation packages
([`../../spherical_gpu`](../../spherical_gpu),
[`../../dictionary_gpu`](../../dictionary_gpu), and the top-level
`kikuchipy`/`orix`/`diffsims` analysis modules). A route should stay thin —
parse the request, call one of these services, serialize the result — while the
heavy lifting (session state, file I/O, image encoding, orientation math,
forward-simulation diagnostics, batch orchestration, crystal-phase lookups)
lives here so it can be unit-tested in isolation and shared across endpoints.

## Key files

### Session & shared state
| File | Description |
|------|-------------|
| [`h5_session.py`](h5_session.py) | Singleton session for the one open HDF5/H5OINA file plus an LRU pattern cache; thread-safe open/close with a bounded deferred-close backlog so concurrent reads never lose their file handle. |
| [`calibration_store.py`](calibration_store.py) | Single source of truth for per-dataset detector geometry and pattern center (PC); derived datasets inherit calibration, PC Refinement writes it, Indexing reads it. |
| [`phase_map_store.py`](phase_map_store.py) | In-memory state + deterministic colour palette for the EDS phase-map builder; persists a `.phase_map.npz` sidecar next to the EBSD file and auto-reloads it on open. |
| [`reference_frame_state.py`](reference_frame_state.py) | Per-file coordinate-system (`R_user`) and pole-figure plotting convention; render-time only, never mutates `xmap.rotations`. |
| [`state_version.py`](state_version.py) | Global monotonic state-version counter; polled by the detached pole-figure window to know when to refetch. |
| [`user_config_manager.py`](user_config_manager.py) | Per-machine user config (API keys, server config, manual paths) in the OS user-config dir; migrates legacy project-tree config files on first load. |

### Imaging & serialization
| File | Description |
|------|-------------|
| [`image_utils.py`](image_utils.py) | Convert 2D NumPy arrays to Base64 PNG (matplotlib-Agg, colormap support); materializes dask-backed lazy arrays first. |
| [`numpy_json.py`](numpy_json.py) | Registers NumPy scalar/array encoders in FastAPI's `ENCODERS_BY_TYPE` once at startup so numpy types serialize cleanly instead of throwing 500s. |

### Orientation & reference frames
| File | Description |
|------|-------------|
| [`orientation_frame.py`](orientation_frame.py) | Vendor-general reference-frame correction; maps Oxford/EDAX/Bruker orientations into one canonical (MTEX-default) frame and measures residual frame offsets. |
| [`orientation_ground_truth.py`](orientation_ground_truth.py) | Loads a vendor's own stored indexing solution (Euler, phase id, grid shape) from an EBSD file for use as ground truth. |
| [`pole_figure.py`](pole_figure.py) | Backend pole-figure renderer (matplotlib-Agg + orix symmetry); manual equal-area/stereographic projection with optional density contours. |
| [`pc_utils.py`](pc_utils.py) | Pattern-center deviation helpers (e.g. PC delta as a percentage) for dictionary indexing. |

### Forward simulation, rendering & refinement
| File | Description |
|------|-------------|
| [`sht_pattern_renderer.py`](sht_pattern_renderer.py) | HTTP-layer wrapper around the SHT `PatternRenderer`; singleton renderer + module-level LRU phase cache for `pattern-match` requests; fail-loud `SHTRenderError` on missing/invalid SHT. |
| [`forward_diagnostics.py`](forward_diagnostics.py) | On-demand forward-simulation diagnostics for an indexing result (forward-NCC, local-anomaly, PC-sensitivity, pattern-residual maps + summary stats). |
| [`refinement.py`](refinement.py) | Joint R+PC refinement service: per-pixel Levenberg–Marquardt (Stage 1), sparse smoothness solve (Stage 2), R-only refine (Stage 3). |
| [`refinement_math.py`](refinement_math.py) | Pure math primitives for refinement: quaternion ↔ so(3) exponential map, disorientation in degrees, sparse 4-neighbour graph Laplacian. |
| [`gpu_sim_runner.py`](gpu_sim_runner.py) | The self-contained "Ours" forward-sim engine — hardware-adaptive Monte-Carlo → master → SHT pipeline that needs no EMsoft/WSL (GPU cupy → numba → PyTorch fallback chain). |
| [`sht_sphere.py`](sht_sphere.py) | Reconstructs an EMSphInx `.sht` master pattern onto a sphere grid (reusing the indexer's inverse-SHT) for the Database Browser's 3D viewer. |
| [`sht_provenance.py`](sht_provenance.py) | Assembles the provenance + parameters document for one `.sht` (File Info panel); sidecar-first, then recovers from `.sht`/`.xtal`/linked CIF without inventing fields. |
| [`engine_router.py`](engine_router.py) | Auto-routes a forward-sim job to the EMsoft or GPU engine using a measured heuristic (reflection count, point-group order, EMsoft availability). |
| [`result_exporter.py`](result_exporter.py) | Builds rich H5 result files (original data + `/Indexing` groups) and exports MTEX-compatible `.ang`/`.ctf` after batch indexing. |

### EDS chemistry & single-pixel phase test
| File | Description |
|------|-------------|
| [`eds_pixel_chemistry.py`](eds_pixel_chemistry.py) | Per-pixel EDS atomic-% for the active dataset; fails soft (returns `None`) so chemistry weighting degrades to pattern-only ranking. |
| [`phase_test.py`](phase_test.py) | Pure assembly logic for the Single-Pixel Phase Test — EDS pre-filtering of library phases and combined ranking (no GPU, no I/O). |
| [`cif_phase_library.py`](cif_phase_library.py) | Builds a per-CIF phase library with normalized atomic-% compositions (via pymatgen) from `Database/crystal_database.xlsx` for EDS phase suggestion. |

### Crystal Hint (symmetry / lattice / phase lookup)
| File | Description |
|------|-------------|
| [`crystal_hint_symmetry.py`](crystal_hint_symmetry.py) | Image-space rotation-NCC symmetry detection around the detected zone axis (fast v1). |
| [`crystal_hint_spherical.py`](crystal_hint_spherical.py) | Method B — spherical-projection symmetry detection (projects detector intensity onto the sphere, rotates around 3D axes) for off-axis robustness. |
| [`crystal_hint_lattice.py`](crystal_hint_lattice.py) | Lattice-parameter estimator: Hough band detection → d-spacings → crystal system + lattice `a` (cubic-focused v1). |
| [`crystal_hint_phase_fit.py`](crystal_hint_phase_fit.py) | Per-phase fit scorers (symmetry-fit and d-spacing-fit) that discriminate among phases passing the categorical filters; also chemistry-fit helpers used by the phase test. |
| [`crystal_hint_region.py`](crystal_hint_region.py) | Region / whole-scan analysis (Mode 2): aggregates symmetry + lattice across sampled pixels, reports histograms and "missing phases". |
| [`crystal_hint_quality.py`](crystal_hint_quality.py) | Quality Check (Mode 3): flags pixels where the detected symmetry is incompatible with the indexed phase's crystal system. |
| [`crystal_hint_presets.py`](crystal_hint_presets.py) | Loads the material presets JSON at import time and exposes query helpers. |
| [`crystal_hint_material_presets.json`](crystal_hint_material_presets.json) | Data file backing `crystal_hint_presets.py` (known material → expected symmetry/lattice/chemistry presets). |
| [`crystal_hint_local_library.py`](crystal_hint_local_library.py) | Indexes the local `CIF_Library` / `XTAL_Library` / `EBSD_SHT_Database` into a unified, queryable phase index. |
| [`crystal_hint_cod.py`](crystal_hint_cod.py) | Crystallography Open Database (COD) client — keyless HTTP queries with 7-day on-disk cache. |
| [`crystal_hint_mp.py`](crystal_hint_mp.py) | Materials Project client (plain `requests`, key from `user_config_manager`); returns `[]` gracefully when no key is configured. |
| [`crystal_hint_cif_downloader.py`](crystal_hint_cif_downloader.py) | Downloads + validates (pymatgen) a CIF from a COD entry into `Database/CIF_Library/`; SHT generation is deferred to the Simulation page. |

### Batch indexing pipeline
| File | Description |
|------|-------------|
| [`batch_manager.py`](batch_manager.py) | Orchestrates batch indexing with a load → preprocess → index → unload cycle per file, wiring together the queue, checkpoint writer, memory guardian and pre-flight checker. |
| [`batch_queue.py`](batch_queue.py) | SQLite-backed persistent job queue (one job = one file × one phase), grouped by file for smart scheduling. |
| [`checkpoint_writer.py`](checkpoint_writer.py) | Writes/appends per-phase results into `_multiphase.h5` checkpoint files for crash recovery. |
| [`memory_guardian.py`](memory_guardian.py) | RAM monitoring + OOM prevention (psutil), estimating in-memory size from on-disk size × decompression factor. |
| [`preflight_check.py`](preflight_check.py) | Pre-flight validation of batch jobs (disk space, memory headroom, etc.) producing a pass/warn/fail report. |

## How this fits the Orienta architecture

```
Electron shell  ─►  React frontend  ─►  FastAPI backend
                                            ├─ api/routes/     (thin HTTP handlers)
                                            ├─ api/services/   ◄── THIS DIRECTORY
                                            └─ spherical_gpu / dictionary_gpu / kikuchipy …
```

Routes in [`../routes`](../routes) import these services to do their work; the
services in turn call the GPU/forward-sim packages
([`../../spherical_gpu`](../../spherical_gpu),
[`../../dictionary_gpu`](../../dictionary_gpu)) and the scientific stack. Keeping
state and logic here (rather than in routes) is what lets the same helper back
several endpoints — e.g. the open-file session, the calibration store and the SHT
renderer are each shared across many routes.

## Run / use notes

- **Not a CLI.** These modules are imported by the running backend, not executed
  directly. Start the backend from the project root:
  `python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000`.
- **In-memory state resets on restart.** `h5_session`, `calibration_store`,
  `phase_map_store`, `state_version`, and the SHT/forward-sim caches live in the
  backend process only — restarting the backend clears them, so an EBSD file must
  be re-loaded after every restart.
- **Startup hook.** `numpy_json.register_numpy_encoders()` must be called exactly
  once at app startup (from `backend.api.main`); without it, responses containing
  NumPy scalars/arrays fail to serialize.
- **One file at a time.** The session model assumes a single open dataset;
  switching files re-loads from disk and resets dependent caches.
- **External services degrade gracefully.** COD has no key; Materials Project
  reads its key from `user_config_manager` and returns empty results when none is
  set. Network lookups are cached on disk under `.cache/crystal_hint/` (7-day TTL).
- **Fail-loud rendering.** The SHT renderer raises `SHTRenderError` on missing
  files / OOM / bad metadata rather than substituting placeholders; routes turn
  these into explicit error responses.
- **Tests.** Each service is exercised by the suite under
  [`../../../tests`](../../../tests); run with `pytest tests/ -v`.
