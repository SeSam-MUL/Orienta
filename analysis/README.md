# `analysis/` — MTEX-equivalent EBSD post-processing

This package performs post-indexing analysis of EBSD data: grain reconstruction,
grain-size statistics, deformation metrics, texture quantification, and
recrystallization classification. It is a pure-Python reimplementation of common
MTEX (MATLAB) microstructure workflows, built on top of
[`orix`](https://orix.readthedocs.io/), NumPy, SciPy, scikit-learn, and
openpyxl. Its single input is an `orix.crystal_map.CrystalMap` produced by any
indexing method (Hough, Dictionary, or Spherical), and its outputs are
statistical result objects, matplotlib-ready arrays, and a multi-sheet
`Documentation.xlsx` report. The package contains **no GUI code** — it is the
scientific layer behind the application's "EBSD Analysis" features.

## Files

| File | Description |
|------|-------------|
| [`__init__.py`](__init__.py) | Public API surface — re-exports every analysis function and result dataclass listed below. |
| [`ebsd_dataset.py`](ebsd_dataset.py) | `EBSDDataset` (a thin adapter that normalizes quality metrics, phases, symmetry, and indexed-pixel masks across indexing methods) and `GrainSet` (per-pixel grain labels plus per-grain properties). The common data model every other module consumes. |
| [`smoothing.py`](smoothing.py) | Orientation pre-smoothing filters (median, mean, quality-weighted mean) applied before grain reconstruction to suppress noise while preserving boundaries. |
| [`grain_analysis.py`](grain_analysis.py) | Misorientation-based grain reconstruction (connected-component segmentation), per-grain properties (mean orientation, GOS), and grain-size analysis (ECD, MaxDim, aspect ratio, area-weighted stats). Returns `GrainSizeResult`. |
| [`bc_analysis.py`](bc_analysis.py) | Band-contrast histogram, 3-Gaussian Mixture Model fit (deformed / recovered / recrystallized populations), and grain-average BC. Returns `BCHistogramResult` / `BCGMMResult`. |
| [`deformation_analysis.py`](deformation_analysis.py) | Kernel Average Misorientation (KAM), grain-boundary classification (SAGB/HAGB), GB length metrics and segment histograms, and grain sphericity. Returns `DeformationResult`. |
| [`texture_components.py`](texture_components.py) | Phase-agnostic ideal texture-component presets (FCC rolling = 14 components, BCC rolling) defined by `{hkl}<uvw>` or Euler angles, with helpers to list, validate, and add custom components. |
| [`texture_analysis.py`](texture_analysis.py) | Texture-component volume fractions, surface plane fractions ({111}/{100}/{110}), and ODF-based texture index / entropy. Returns `TextureResult`. |
| [`rx_analysis.py`](rx_analysis.py) | Recrystallization classification using GOS / gBC / gKAM criteria, RX area fraction, and area-weighted GOS/gKAM histograms. Returns `RXResult`. |
| [`excel_exporter.py`](excel_exporter.py) | Writes a 14-sheet `Documentation.xlsx` (basic info, BC histogram + GMM, grain-size / KAM / GOS / gKAM histograms, grain boundaries, texture components, surface planes, RX metrics), one column per dataset. |
| [`batch_processor.py`](batch_processor.py) | Scans a folder of EBSD files (`.h5`, `.h5oina`, `.ang`, `.ctf`), runs the full pipeline on each, and consolidates everything into a single `Documentation.xlsx`. |

## How it fits the Orienta architecture

This is the scientific post-processing layer of the
[Electron + React + FastAPI](../docs/ARCHITECTURE.md) stack:

```
React frontend  →  FastAPI backend  →  analysis/  →  orix / numpy / scipy / sklearn
```

The backend route module [`backend/api/routes/analysis.py`](../backend/api/routes/analysis.py)
imports from this package and exposes the analysis to the UI over HTTP. Indexing
results from the various indexing backends (e.g. `backend/spherical_gpu/`,
the Dictionary path, and Hough indexing) all converge on a single
`orix.crystal_map.CrystalMap`, which `EBSDDataset` then normalizes so the rest of
the pipeline is indifferent to how the map was produced.

Typical flow within the package:

```
CrystalMap → EBSDDataset → (smoothing) → reconstruct_grains → GrainSet
          → analyze_grain_size / analyze_deformation / analyze_texture
          → analyze_recrystallization / BC histogram + GMM
          → ExcelExporter / batch_process_folder → Documentation.xlsx
```

## Use notes

- **Pure library, no PyQt5 / Electron / FastAPI here.** Import it directly:
  ```python
  from analysis import EBSDDataset, reconstruct_grains, analyze_grain_size
  ```
- **Optional dependencies are imported defensively.** Modules guard `orix`,
  `scipy`, `scikit-learn`, and `openpyxl` behind `*_AVAILABLE` flags. The full
  feature set requires `orix`, `numpy`, `scipy`, `scikit-learn`, `openpyxl`, and
  `matplotlib` (see the project [`requirements.txt`](../requirements.txt)).
- **Input contract:** functions operate on an `EBSDDataset` wrapping an
  `orix` `CrystalMap`. Multi-match maps (Dictionary indexing with `keep_n > 1`)
  are automatically reduced to the best match per pixel.
- **Tests** live alongside the project test suite, e.g.
  `tests/test_ebsd_analysis.py`, `tests/test_grain_analysis`-related files,
  `tests/test_bc_analysis.py`, `tests/test_texture_components.py`,
  `tests/test_excel_exporter.py`, and `tests/test_batch_processor.py`. Run them
  with `pytest tests/ -v`.
- The MTEX scripts these modules reproduce, and the architecture mapping, are
  documented in the MTEX-equivalence design notes and the `matlab_testskripts/`
  reference directory at the project root.
