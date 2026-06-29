# `backend/dict_gpu` — GPU dictionary indexing

GPU-accelerated **dictionary indexing** for EBSD: given a kikuchipy master
pattern (or a pre-generated dictionary) and a set of experimental patterns,
this package forward-projects the master onto the detector for a fundamental-zone
sampling of orientations, then finds the best-matching orientation per pixel by
normalised cross-correlation (NCC) on the GPU. It returns an
[`orix`](https://orix.readthedocs.io/) `CrystalMap` wrapped in an
`IndexingResult` that is layout-compatible with the project's CPU dictionary
path, so downstream modules (Phase Map, EBSD Analysis, exporters) consume it
unchanged. The whole pipeline runs in PyTorch on CUDA, with VRAM-aware tiling,
optional PCA dimensionality reduction, and fail-loud contract errors.

## Attribution

Two third-party-derived components live here; both are recorded in the
repository-level [`../../NOTICE.md`](../../NOTICE.md):

- **Vendored MIT pcadi core** under [`_pcadi/`](_pcadi/) — PCA, GEMM top-k
  k-NN, INT8 quantization, and the master→detector projection, adapted from the
  [`pcadi`](https://github.com/ZacharyVarley/pcadi) project (commit `88f676e`,
  MIT). Provenance and per-symbol modifications are in
  [`_pcadi/__SOURCE.md`](_pcadi/__SOURCE.md). The MIT copyright notice is
  retained in the source headers there.
- **kikuchipy-derived projection math** under
  [`_pcadi/_projection/`](_pcadi/_projection/) — direction cosines, square
  Lambert mapping, and bilinear sampling, **translated** (not copied) into
  vectorised PyTorch from kikuchipy 0.11.3's `_master_pattern.py`. See
  [`_pcadi/_projection/__SOURCE.md`](_pcadi/_projection/__SOURCE.md). Because
  this math is derived from GPL-3.0 kikuchipy, the combined work is GPL-3.0-or-later
  (see [`../../NOTICE.md`](../../NOTICE.md)).

## Layout

| Path | Description |
|------|-------------|
| [`__init__.py`](__init__.py) | Public exports: `gpu_dictionary_index_patterns`, `detect_gpu`, `GpuStatus`, and the error types. |
| [`api.py`](api.py) | Thin public entry point — re-exports `pipeline.indexer.run_dictionary_index` as `gpu_dictionary_index_patterns`. |
| [`runtime.py`](runtime.py) | CUDA detection (`detect_gpu` → `GpuStatus`) and VRAM budgeting (`vram_budget_bytes`). Imports torch lazily so torch-less / CPU-only environments don't crash. |
| [`exceptions.py`](exceptions.py) | Contract errors surfaced verbatim to the UI: `GpuDictError`, `VramExhaustedError`, `MasterPatternError`. |
| [`pipeline/`](pipeline/) | End-to-end orchestration (see below). |
| [`_pcadi/`](_pcadi/) | Vendored MIT pcadi core + kikuchipy-derived projection (see below). |

### `pipeline/` — orchestration

| Path | Description |
|------|-------------|
| [`pipeline/indexer.py`](pipeline/indexer.py) | The orchestrator (`run_dictionary_index`): resolves master/dict input, samples orientations, generates the dictionary on GPU, applies optional PCA, normalises, runs tiled top-k NCC, builds the `CrystalMap`, releases VRAM, and emits per-phase timing diagnostics. |
| [`pipeline/master_loader.py`](pipeline/master_loader.py) | Loads a kikuchipy `EBSDMasterPattern`, a kikuchipy-written dictionary signal, or a raw legacy dict-cache `.h5`, returning a tagged `MasterPayload` (carries `phase` for IPF colouring where available). |
| [`pipeline/grid.py`](pipeline/grid.py) | Fundamental-zone orientation sampling at a given angular step, wrapping `orix.sampling.get_sample_fundamental`. |
| [`pipeline/tiling.py`](pipeline/tiling.py) | Pure-CPU VRAM-aware tiling math (`compute_tile_size`, `iter_tiles`) so the dictionary is matched in slices that fit the budget. |
| [`pipeline/output.py`](pipeline/output.py) | Back-maps the top-k GPU results onto the original grid coordinates and builds an `orix` `CrystalMap` (matches the CPU path's layout). |

### `_pcadi/` — vendored MIT core + derived projection

| Path | Description |
|------|-------------|
| [`_pcadi/__SOURCE.md`](_pcadi/__SOURCE.md) | Provenance for the pcadi vendoring: source commit, MIT license text, extracted symbols, and modifications. |
| [`_pcadi/pca.py`](_pcadi/pca.py) | `GpuPCA` — SVD-based PCA on GPU tensors (fit / transform / inverse_transform). |
| [`_pcadi/knn.py`](_pcadi/knn.py) | `gemm_topk_ncc` — single-GEMM top-k NCC search (FP16 GEMM, FP32 top-k); caller supplies mean-centred, L2-normed inputs. |
| [`_pcadi/quantize.py`](_pcadi/quantize.py) | Per-row INT8 quantize/dequantize round-trip helpers (VRAM-saving option). |
| [`_pcadi/master_to_dict.py`](_pcadi/master_to_dict.py) | `gpu_master_to_dict` — generates dictionary patterns on GPU via real forward projection (**Path A**, default), with a CPU `mp.get_patterns()` wrapper retained as the `_use_path_b=True` escape hatch. |
| [`_pcadi/_projection/__SOURCE.md`](_pcadi/_projection/__SOURCE.md) | Translation notes mapping each function to its kikuchipy 0.11.3 reference symbol and lines. |
| [`_pcadi/_projection/direction_cosines.py`](_pcadi/_projection/direction_cosines.py) | Detector-pixel → 3D sample-frame direction cosines (vectorised PyTorch). |
| [`_pcadi/_projection/lambert.py`](_pcadi/_projection/lambert.py) | Vector ↔ square Lambert mapping. |
| [`_pcadi/_projection/sample.py`](_pcadi/_projection/sample.py) | Bilinear sampling of the master hemispheres via `grid_sample`. |
| [`_pcadi/_projection/project.py`](_pcadi/_projection/project.py) | Composes rotate + Lambert + sample into the full master→detector forward projection, vectorised over all rotations. |

## How it fits into Orienta

This package is the **GPU compute backend for dictionary indexing**, sitting in
the scientific layer beneath the FastAPI/React stack:

```
IndexingPage (React)  →  routes/indexing  →  indexing_controller.py
                                                   │  (compute mode = "gpu")
                                                   ▼
                              backend/dict_gpu.api.gpu_dictionary_index_patterns
```

`indexing_controller.py` routes dictionary indexing here when the user selects
GPU compute (it first checks `detect_gpu()` and falls back / fails loud as
configured); otherwise it stays on the kikuchipy CPU path. The result type and
`CrystalMap` layout intentionally match the CPU path so the rest of the app does
not branch on which backend produced the map. GPU device status is also exposed
through the system route ([`../api/routes/system.py`](../api/routes/system.py))
via `detect_gpu()`.

> **Not to be confused with** the sibling
> [`../dictionary_gpu/`](../dictionary_gpu/) package, which is the standalone
> "Dictionary (GPU)" dictionary-*generation* tool (its own route and page). This
> `dict_gpu` package is the indexing pipeline wired into `indexing_controller`.
> See also the spherical-indexing GPU backend in
> [`../spherical_gpu/`](../spherical_gpu/).

## Use / run notes

- **Entry point.** Call `gpu_dictionary_index_patterns(experimental_signal,
  master_pattern_or_path, detector, ...)`. The second argument may be a path, a
  kikuchipy `EBSDMasterPattern`, or a kikuchipy EBSD signal carrying
  pre-generated dictionary patterns; see the docstring in
  [`pipeline/indexer.py`](pipeline/indexer.py) for the full keyword list
  (`angular_step_deg`, `keep_n`, `selection_mask`, `use_pca`, `pca_components`,
  `vram_budget_gb`, `progress_callback`, `cancel_check`).
- **Requires CUDA.** If no CUDA device is available the call raises
  `GpuDictError` rather than silently using the CPU — choose the CPU path
  upstream instead. `detect_gpu()` / `runtime.py` are safe to import without a
  GPU (torch is imported lazily).
- **VRAM management.** PCA is auto-enabled under memory pressure or for large
  selections; matching is tiled to the VRAM budget; the large GPU tensors are
  explicitly freed after the `CrystalMap` is built so the next run starts with a
  full budget. Override the budget with `vram_budget_gb`.
- **Fail-loud contracts.** Detector/experimental shape mismatches, non-finite
  dictionaries, empty selections, and missing phase `point_group` (needed for
  IPF colouring) raise explicit errors.
- **Only the NCC metric** is implemented (`metric="ncc"`).
- **Tests** live in [`../../tests/test_dict_gpu/`](../../tests/test_dict_gpu/),
  including CPU↔GPU parity, projection (Path A) correctness, and tiling/PCA/k-NN
  unit tests.
