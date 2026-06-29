# `backend/workers`

This directory is a **reserved Python package** for Orienta's background worker
modules — the code that drives long-running jobs (batch indexing, master-pattern
simulation, GPU compute) outside the request/response cycle of the FastAPI API.

At present the package is a placeholder: it contains only an empty
[`__init__.py`](__init__.py) so that `backend.workers` is importable. The
background-job logic it is intended to host currently lives next to the API, in
[`../api/services/`](../api/services) and the dedicated compute packages. This
README documents where that work happens today and what this directory is meant
to consolidate over time, so contributors don't search for a worker that isn't
here yet.

## Contents

| File / subdir | Description |
| --- | --- |
| [`__init__.py`](__init__.py) | Empty package marker. Makes `backend.workers` an importable namespace; no symbols are exported yet. |

There are no other files or subdirectories in this directory.

## Where long-running jobs actually run today

Until worker modules are added here, the heavy/long-running work is handled by
service modules under [`../api/services/`](../api/services):

- **Batch indexing queue** — [`../api/services/batch_queue.py`](../api/services/batch_queue.py):
  a SQLite-backed persistent job queue (one job = one file × one phase), so a
  batch survives a backend restart.
- **Batch orchestration** — [`../api/services/batch_manager.py`](../api/services/batch_manager.py):
  loads → preprocesses → indexes → unloads each file in turn, with memory
  safety and checkpointing.
- **Crash-safe checkpoints** — [`../api/services/checkpoint_writer.py`](../api/services/checkpoint_writer.py):
  periodically persists partial results during a batch.
- **Memory safety** — [`../api/services/memory_guardian.py`](../api/services/memory_guardian.py)
  and [`../api/services/preflight_check.py`](../api/services/preflight_check.py):
  gate large jobs before/while they run.
- **GPU simulation runner** — [`../api/services/gpu_sim_runner.py`](../api/services/gpu_sim_runner.py):
  drives GPU-side master-pattern / forward work.

The HTTP entry points that launch and poll these jobs are the batch and
simulation routes under [`../api/routes/`](../api/routes) (for example
`batch_v2.py` and `simulation.py`).

## How it fits the Orienta architecture

Orienta is an Electron + React front end talking over HTTP/WebSocket to a
FastAPI back end on the kikuchipy / orix / diffsims scientific stack:

```
Electron + React  ──HTTP/WebSocket──▶  FastAPI (backend/api)
                                            │
                                            ├── backend/api/services   ← job orchestration today
                                            ├── backend/workers        ← reserved for worker modules (this dir)
                                            ├── backend/spherical_gpu   ← spherical-indexing GPU compute
                                            ├── backend/dictionary_gpu  ← dictionary-indexing GPU compute
                                            └── backend/forward_sim     ← forward master-pattern simulation
```

The intent of `backend/workers` is to give those background-job concerns a home
of their own, separate from the request-handling services in
[`../api/`](../api) and from the numerical GPU packages
([`../spherical_gpu`](../spherical_gpu),
[`../dictionary_gpu`](../dictionary_gpu),
[`../forward_sim`](../forward_sim)).

## Run / use notes

- Nothing here is executable on its own — the package is empty and exports no
  symbols. There is no worker process to start from this directory.
- The back end has **no auto-reload**: any Python change requires killing and
  restarting the backend process. New worker modules added here are picked up
  only on restart.
- When adding a worker module, keep it scoped to this package and import shared
  helpers from [`../api/services/`](../api/services) rather than duplicating the
  queue / checkpoint / memory-guard logic that already exists there.
