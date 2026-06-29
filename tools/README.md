# `tools/` — Standalone data utilities

This directory holds **pure data-logic helpers** that have no GUI or web-framework
dependencies. Each module reads or processes EBSD/EDS data (HDF5 patterns, indexing
results, phase maps) using only NumPy, h5py, orix, and matplotlib. Because they are
free of UI code, the same functions can be called from the FastAPI backend, from the
test suite, or directly in a Python/Jupyter session. The most prominent member is the
**HDF5 viewer backend**, which extracts patterns, EDS spectra, electron images, and
indexing metadata from Oxford H5OINA and EDAX HDF5 files with lazy, cached loading.

## Files

| File | Description |
|------|-------------|
| [`h5_viewer_backend.py`](h5_viewer_backend.py) | `H5OINADataExtractor` — reads Oxford H5OINA and EDAX HDF5 files. Lazy single-pattern loading with a small FIFO cache, grid/pattern-shape detection, EDS spectra & element maps, electron images (SE/BSE/FSE), Band Contrast, per-pixel Aztec records, EBSD/EDS header metadata, on-the-fly IPF map computation via orix, and a `enumerate_layers()` catalog of everything present in a given file. Fails loud on corrupt headers (e.g. non-positive grid dimensions) rather than silently clamping. |
| [`pattern_comparison.py`](pattern_comparison.py) | NCC-based pattern-quality utilities for dictionary/Hough/spherical indexing results: per-pixel NCC score maps, best-match simulated pattern retrieval (incl. lazy single-pattern fetch from a released GPU dictionary file via `simulation_indices`), experimental-pattern retrieval, element-wise NCC images and scalars, circular-aperture (EDAX-style) detection/masking, and a base64-PNG NCC difference renderer. |
| [`phase_map_generator.py`](phase_map_generator.py) | Publication-quality phase-map and IPF rendering with matplotlib (headless `Agg` backend). Builds `PhaseMapData` from an orix `CrystalMap` or a raw 2D phase-ID array, computes IPF colours per phase (deriving the point group from the space group when needed), and renders/exports figures with a configurable `ScaleBarConfig` and legend. |
| [`__init__.py`](__init__.py) | Marks the directory as a Python package (`tools`). |

> `__pycache__/` contains compiled bytecode and can be ignored.

## How this fits the Orienta architecture

These modules sit in the **scientific/data-logic layer**, below the FastAPI backend
and well below the React/Electron frontend. They are imported by backend routes and
services rather than called directly by the UI. For example:

- [`backend/api/routes/h5_viewer.py`](../backend/api/routes/h5_viewer.py) and
  [`backend/api/services/h5_session.py`](../backend/api/services/h5_session.py) wrap
  `H5OINADataExtractor` to expose HDF5 viewing endpoints.
- [`backend/api/routes/indexing.py`](../backend/api/routes/indexing.py) and the
  forward-diagnostics service use `pattern_comparison` for the Pattern Match dialog
  and NCC heatmaps.
- [`backend/api/routes/phase_map.py`](../backend/api/routes/phase_map.py) and
  [`backend/api/routes/analysis.py`](../backend/api/routes/analysis.py) use
  `phase_map_generator` for phase/IPF map rendering and export.

Keeping this logic UI-free is deliberate: it makes the functions reusable across the
API, the test suite, and ad-hoc scripts, and keeps GUI changes from touching the
data path. See the [project README](../README.md) for the overall architecture.

## Use notes specific to this directory

- **Import as a package**, e.g. `from tools.h5_viewer_backend import H5OINADataExtractor`,
  with the project root on `sys.path` (the standard way the backend and tests run).
- **No GUI / web dependencies here** — these are plain Python modules. Adding PyQt or
  FastAPI imports to this directory breaks the layering contract.
- **`H5OINADataExtractor` takes an already-open `h5py.File`** plus a format string
  (`'Oxford'` or `'EDAX'`); it does not open or close files itself. Caller owns the
  file handle's lifetime.
- **Fail-loud conventions** are intentional: corrupt grid headers, shape mismatches
  between header and data, and phases with no derivable symmetry raise `ValueError`
  instead of returning misleading defaults.
- **`phase_map_generator` uses the non-interactive `Agg` matplotlib backend** for
  headless rendering and requires the optional `matplotlib-scalebar` dependency.
- Relevant tests live in [`../tests/`](../tests/) (e.g. `test_phase_map_generator.py`,
  `test_pattern_comparison_mask.py`, `test_ncc_diff_png.py`).
