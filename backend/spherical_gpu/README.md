# `backend/spherical_gpu` — GPU Spherical (SHT) Indexing

This package is Orienta's GPU-accelerated **spherical-harmonic-transform (SHT)
indexing** backend for EBSD patterns. It mirrors the option surface of the
EMSphInx `IndexEBSD` pipeline but runs on the GPU through PyTorch: each
experimental pattern is preprocessed, projected onto a square-Legendre grid,
transformed to spherical-harmonic coefficients, cross-correlated against a `.sht`
master pattern over all of SO(3), and decoded to an orientation. It supports
single- and multi-phase runs, hardware-adaptive batching with OOM recovery, a
forward pattern renderer, and a dedicated **pseudo-symmetry resolver** for
cubic-approximant intermetallics. The detector/Lambert projection math is derived
from **kikuchipy (GPL-3.0)** — see [`../../NOTICE.md`](../../NOTICE.md) for the
full attribution and the resulting GPL-3.0-or-later license of the combined work.

> **Cite the method.** This is an **independent reimplementation** of the
> spherical-harmonic indexing method of **Lenthe, Singh & De Graef,
> _Ultramicroscopy_ 207, 112841 (2019)** (as realized in the **EMSphInx** project,
> Carnegie Mellon University). It contains no EMSphInx source code. Please cite that
> paper and EMSphInx in any publication that uses this backend. See
> [`../../NOTICE.md`](../../NOTICE.md) → "Algorithm / method attributions".

## Top-level files

| File | Description |
|------|-------------|
| [`backend.py`](backend.py) | Production entry point: `SphericalGPUBackend`, `BackendConfig`, `PhaseConfig`. Single facade upstream callers (FastAPI routes, batch jobs) use; handles device selection, lazy per-phase indexer construction, batch-level multi-phase interleaving, streaming H5/array/signal/dataset inputs, and cooperative cancellation. |
| [`runtime.py`](runtime.py) | GPU runtime detection, memory-aware batch sizing (`compute_safe_batch`), and OOM-retry recovery (`run_with_oom_retry`). Makes the pipeline robust across 4 GB laptops to 12 GB+ workstations and bandwidths L=53–113. |
| [`pseudosym.py`](pseudosym.py) | Pseudo-symmetry quaternion math for cubic-approximant point groups (`23`, `m-3`): symmetry cosets, variant quaternions, and same-orientation/disorientation metrics used by the resolver. |
| [`eds_phase_filter.py`](eds_phase_filter.py) | Optional EDS-aware phase pre-filter: builds a per-pixel × per-phase compatibility mask from H5OINA EDS chemistry to skip incompatible phases during multi-phase indexing. Conservative — falls back to "all phases compatible" when EDS data is missing. |
| [`exceptions.py`](exceptions.py) | Custom exception hierarchy (`SphericalGPUError` base plus `SHTReadError`, `DetectorConventionError`, `GPUUnavailableError`, `OracleIntegrityError`). |
| [`__init__.py`](__init__.py) | Package surface; re-exports the exception types. |

## `pipeline/` — EBSD-specific stages

Orienta's own implementation of the indexing stages (the math primitives live in
`_math/`). See [`pipeline/__init__.py`](pipeline/__init__.py).

| File | Description |
|------|-------------|
| [`pipeline/indexer.py`](pipeline/indexer.py) | `Tier1Indexer` + `IndexResult`: the core single-pass SHT cross-correlation indexer (square-Legendre grid SHT → SO(3) correlation volume → peak → Euler angles). |
| [`pipeline/detector.py`](pipeline/detector.py) | `DetectorGeometry` and Pattern-Center convention conversions (Oxford / EDAX / EMsoft / Bruker → EMsoft internal units). |
| [`pipeline/preprocessing.py`](pipeline/preprocessing.py) | GPU-batched pattern preprocessing matching EMSphInx ordering: `circmask` → `gausbckg` → `nregions` (adaptive histogram equalization). |
| [`pipeline/refiner.py`](pipeline/refiner.py) | Tier-2 refinement: 3D Gaussian smoothing of the cc volume + sub-bin parabolic peak interpolation (the `refine` option). |
| [`pipeline/resolution.py`](pipeline/resolution.py) | Pseudo-symmetry orientation resolution for `m-3`/`23` phases. Production path replaces confused variants with Hough band-geometry orientations (`resolve_map`/`resolve_eulers`); the coset top-K machinery is retained as a tested non-default alternative. |
| [`pipeline/forward.py`](pipeline/forward.py) | Forward EBSD pattern renderer: `.sht` → inverse SHT → Driscoll-Healy grid → sampled simulated 2D pattern (used for render-NCC validation and Pattern Match). |
| [`pipeline/sht_io.py`](pipeline/sht_io.py) | Reader for EMsoft `.sht` binary master-pattern files (self-contained binary format, not HDF5). |
| [`pipeline/output.py`](pipeline/output.py) | Converts an `IndexResult` into an orix `CrystalMap` and writes `.ang`/`.ctf` files. |
| [`pipeline/_shared_tables.py`](pipeline/_shared_tables.py) | `SharedSphericalTables`: L+geometry-only precomputations shared across phases of a same-bandwidth multi-phase run to skip redundant build cost. |

## `_math/` — vendored harmonic primitives

Low-level math primitives (Wigner-d matrices, real/complex SHT, SO(3)
cross-correlation kernels, Lambert/cubochoric projections, fundamental-zone
grids). The core transforms are **vendored from ebsdtorch (MIT)** with imports
patched to relative form; see [`_math/__SOURCE.md`](_math/__SOURCE.md) for exact
provenance, the upstream commit SHA, and the patch log, and
[`_math/LICENSE.ebsdtorch`](_math/LICENSE.ebsdtorch) for the MIT text. Public
surface used by the pipeline is re-exported from [`_math/__init__.py`](_math/__init__.py)
(`rosca_lambert`, `wigner_d`, `RSHT`, `rs2cc_`, etc.). Additional local modules
add fused CUDA kernels, EMSphInx-compatible Wigner evaluation, and inverse/Newton
SHT helpers.

> **Licensing note.** The detector and Lambert projection math used here derives
> from **kikuchipy (GPL-3.0)**, which is why this directory contributes to
> Orienta's GPL-3.0-or-later licensing. The MIT-licensed `_math/` primitives are
> compatible with that combination. Full details: [`../../NOTICE.md`](../../NOTICE.md).

## How this fits into Orienta

```
React frontend ──HTTP──▶ FastAPI backend (backend/api/)
                              │
                              ▼
                    SphericalGPUBackend  (this directory)
                              │
                   ┌──────────┴───────────┐
                   ▼                      ▼
            pipeline/ stages        _math/ primitives
        (detector, preprocess,    (SHT, Wigner-d, Lambert,
         SHT index, refine,        cross-correlation)
         resolve, forward, I/O)
```

`SphericalGPUBackend` is the spherical-indexing path inside Orienta's broader
indexing layer. The FastAPI routes and the higher-level indexing controller call
it to index whole maps or ROIs and receive orientations as an orix `CrystalMap`,
which then feeds phase maps, pattern matching, and the analysis modules. It is a
sibling to:

- `backend/dict_gpu/` — GPU **dictionary** indexing (shares the kikuchipy-derived
  projection math noted in [`../../NOTICE.md`](../../NOTICE.md)),
- the Hough indexing path (PyEBSDIndex), and
- the EMSphInx CPU backend (invoked via WSL).

This directory is self-contained: the `pipeline/` stages are Orienta's own code,
and the harmonic primitives are vendored into `_math/` so the package has no
runtime dependency on an installed `ebsdtorch`.

## Run / use notes

- **Entry point.** Construct from an indexing config and index:
  ```python
  from backend.spherical_gpu.backend import SphericalGPUBackend, BackendConfig
  bcfg = BackendConfig.from_indexing_config(config, sht_files=[...])
  backend = SphericalGPUBackend(bcfg)
  result = backend.index_h5(h5_path, detector_params)   # also index_array / index_signal / index_h5_dataset
  ```
- **Input.** `detector_params` is the same dict EMSphInx callers already pass
  (`n_rows, n_cols, pat_width, pat_height, pixel_size, tilt, binning, step_x,
  step_y, pc_x, pc_y, pc_z, vendor`). Master patterns are EMsoft `.sht` files,
  one per phase.
- **Device.** CUDA is auto-detected; with no GPU it falls back to CPU with a loud
  warning (CPU is far slower — prefer the EMSphInx CPU backend for production
  CPU-only runs).
- **Batch size.** Auto-picked from free GPU memory per bandwidth; OOM halves the
  batch and retries. Override via `BackendConfig.batch_size` if needed.
- **Detector changes.** One indexer set is cached per detector geometry. If the
  detector params change (e.g. PC drift), call `backend.invalidate()` before
  re-indexing.
- **Cancellation.** Install a cooperative-cancel callback with
  `backend.set_cancel_check(fn)`; it is polled once per batch.
- **Pseudo-symmetry.** For cubic-approximant phases (`m-3`/`23`) the raw SHT
  correlation can land on a wrong pseudo-symmetric variant; the resolver in
  [`pipeline/resolution.py`](pipeline/resolution.py) corrects this using Hough
  band-geometry orientations. Variant correctness is judged by render-NCC, not by
  symmetry disorientation.
