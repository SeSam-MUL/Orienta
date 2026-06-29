# `simulation/` — EMsoft master-pattern simulation & file sync

This package is Orienta's bridge to the **EMsoft** and **EMSphInx** Fortran/OpenCL
toolchain, which runs inside **WSL** (or natively on Linux). From a crystal
structure file (`.xtal`) it drives the full EBSD master-pattern pipeline — Monte
Carlo electron scattering, master-pattern computation, and the spherical-harmonic
transform (`.sht`) used by spherical indexing — then keeps the resulting files
synchronized between the local cache and an optional shared network database. The
modules here cover configuration, background execution, system/GPU detection,
upload/download, caching, and the NML templates that parameterize the EMsoft
binaries.

## Key files

| File / dir | Description |
|------------|-------------|
| `simulation_controller.py` | Core business logic. Defines `SimulationParameters`, `SimulationJob`, `SimulationStatus`, `OutputType` (SHT-only / master-only / both) and the `SimulationController` + `BatchQueueManager`. Builds NML files from the templates, launches jobs, parses progress, and provides module-level helpers: `reap_orphaned_emsoft_processes()` (kills WSL EMsoft processes orphaned by a backend restart), `sample_emsoft_cpu_cores()` + `_format_heartbeat()` (liveness heartbeat while `EMEBSDmasterSHT` runs silently), `recommend_dmin()` / `read_xtal_max_lattice_nm()` (adaptive `dmin` per cell size), and `read_xtal_min_occupancy()` (disordered-structure warning). |
| `simulation_worker.py` | `SimulationWorker(QThread)` — runs a single simulation off the UI thread, emitting `progress_update` / `stage_changed` / `simulation_finished` signals. Wraps the EMsoft automation call and forwards its log records to the GUI via `ProgressLogHandler`. |
| `opencl_detector.py` | Parses `clinfo` to enumerate OpenCL platforms/devices, discovers which of the 5 EMsoft binaries (`EMMCOpenCL`, `EMMC`, `EMEBSDmaster`, `EMEBSDmasterOpenCL`, `EMEBSDmasterSHT`) are present, and recommends a GPU-vs-CPU execution path. Computes the EMsoft GPU-only device id (`emsoft_devid`), validates/repairs broken ICD vendor files, and clamps work-group size. |
| `system_check.py` | `SystemStatus` + checks that verify WSL is installed, EMsoft/EMSphInx binaries exist, the config is valid, and OpenCL is usable. Runs short-timeout `wsl bash -lc` probes (login shell, no interactive flag) and enriches results with `opencl_detector`. |
| `file_sync_manager.py` | `FileSyncManager(QObject)` — extracts the dominant material element from a `.xtal` (HDF5 `AtomData`, via the periodic-table map) for foldering, and coordinates uploads of simulation results through `ServerModeManager`. Tracks `UploadRecord` history. |
| `server_mode_manager.py` | `ServerModeManager(QObject)` — event-driven network connectivity checks (no polling), a background upload queue/worker, and online/offline state. Emits `connectivity_changed`, `upload_progress`, `upload_status`. |
| `crystal_sync.py` | `CrystalSyncManager` — bidirectional CIF/XTAL sync on a 5-minute `QTimer`, with hash-based conflict detection (`SyncConflict`, `ConflictAction`). |
| `database_browser.py` | Unified browse model over local cache + server (`BrowserEntry`, `FileLocation`), with on-demand download backed by the LRU cache. |
| `lru_cache.py` | `LRUCacheManager` — local file cache with a size limit (default 20 GB) and least-recently-used eviction for downloaded `.sht` / `.h5` files. |
| `master_pattern_loader.py` | GUI-free loader for master-pattern `.h5`, Monte-Carlo `.h5`, and `.sht` files (`MasterPatternData`, `detect_file_type`). |
| `dictionary_generator.py` | Builds simulated pattern dictionaries from a master pattern (orix orientations → kikuchipy detector projection), with on-disk caching and `DictionaryMetadata` sidecars. |
| `install_emsoft.sh` | Resumable Bash installer for the full WSL/Linux toolchain: swap, build deps, POCL/OpenCL, CMake, EMsoft superbuild, EMSphInx, config + PATH, verification. |
| `templates/` | EMsoft NML input templates (`{...}` placeholders filled by the controller): `EMMCOpenCL.nml.template` (Monte Carlo, GPU), `EMEBSDmaster.nml.template` / `EMEBSDmasterOpenCL.nml.template` (master pattern, CPU/GPU), `EMEBSDmasterSHT.nml.template` (`.sht` for EMSphInx), and `BetheParameters.nml.template`. |
| `__init__.py` | Empty package marker. |

## How this fits the Orienta architecture

Orienta runs a React/Electron frontend over a FastAPI backend on the
kikuchipy / orix / diffsims stack. This package is the **business-logic layer**
that turns crystal structures into the master-pattern / `.sht` files the rest of
the app indexes against:

- The simulation **GUI** triggers jobs that flow through `SimulationController`,
  which spawns a `SimulationWorker` thread; each job shells out to the EMsoft
  binaries inside WSL.
- Produced `.sht` files feed **spherical (EMSphInx) indexing**, while master
  `.h5` files feed **Dictionary indexing** (kikuchipy). The GPU/Dictionary
  indexing code lives in sibling packages — see
  [`../backend/spherical_gpu/`](../backend/spherical_gpu/) and
  [`../backend/dictionary_gpu/`](../backend/dictionary_gpu/).
- `FileSyncManager` + `ServerModeManager` + `crystal_sync.py` + `lru_cache.py` +
  `database_browser.py` keep these outputs (and source CIF/XTAL files) in step
  between the local cache and an optional shared network database.
- Filename/path conventions are centralized in the project-root
  [`../path_utils.py`](../path_utils.py) (`sanitize_filename`, `is_wsl`,
  database path discovery), which these modules import rather than reimplement.

## Run / use notes

- **WSL is required for simulation.** EMsoft/EMSphInx are Fortran/OpenCL binaries
  that run inside the WSL2 VM (default location `~/emsoft/builds/EMsoft-Release/Bin/`).
  Use `install_emsoft.sh` once inside WSL to build the toolchain; it is resumable
  via marker files in `~/.emsoft_install_progress`.
- **The Windows `wsl.exe` relay is not the compute process.** Killing the backend
  only stops the relay — the EMsoft job keeps running in the WSL VM. The backend
  calls `reap_orphaned_emsoft_processes()` on startup to clean up such orphans.
- **GPU vs CPU is auto-detected.** `opencl_detector` / `system_check` pick a path
  and compute EMsoft's GPU-only device id; `EMEBSDmasterSHT` is CPU-only by design.
- **`dmin` matters for large cells.** Reflection count scales with `a_max / dmin`;
  use `recommend_dmin()` so large unit cells don't take days. Strongly disordered
  structures (low site occupancy) can hang EMsoft — `read_xtal_min_occupancy()`
  surfaces a warning.
- **Long silent stages are normal.** `EMEBSDmasterSHT` emits no progress while it
  computes on the Legendre grid; the controller's heartbeat reports elapsed time
  and live CPU-core usage so a running job is not mistaken for a stalled one.
