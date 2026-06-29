# Vendored from ZacharyVarley/pcadi

- **Commit:** 88f676e6f44812900b3c873692aa1a1c9e48b746
- **License:** MIT (see verbatim text below)
- **Imported on:** 2026-05-10
- **Local clone:** `E:/pcadi-reference/`

## Extracted symbols

| File here | Source file | Source symbol(s) | Source lines (approx) | Modifications |
|---|---|---|---|---|
| `pca.py` | `utils.py` | inspired by `OnlineCovMatrix` | 2080–2400 | Replaced streaming accumulator with direct SVD on centered data; renamed `PCA` → `GpuPCA`; removed CPU/NumPy fall-back path; added explicit shape & component-count validation |
| `quantize.py` | `utils.py` (also `utils_knn.py`) | inspired by `LinearLayer` quantization | utils.py 4650–4720, utils_knn.py 8–110 | Trimmed to quantize/dequantize only (dropped quantised linear-forward path); per-row scale + zero-point with constant-row safeguard |
| `knn.py` | `utils.py` (also `utils_knn.py`) | inspired by `ChunkedKNN` GEMM core | utils.py 4720–5651, utils_knn.py 110–298 | Stripped to single-pass GEMM + top-k; chunking moved to pipeline/tiling.py (Task 13); explicit FP16/FP32 toggle |
| `master_to_dict.py` | (none — wrapper around kikuchipy CPU path) | n/a | n/a | **Path B (MVP wrapper)**. Calls `master_pattern.get_patterns(...)` on CPU and uploads the resulting dictionary block to GPU. Replacing this with pcadi's true GPU projection (`MasterPattern` + `EBSDGeometry`, utils.py 3321–4649) is a follow-up; the public API is stable. |

## Modifications from upstream

- Removed CPU/NumPy fall-back path; this module is GPU-by-construction (CPU torch tensors also work but are not optimised).
- Renamed `PCA` → `GpuPCA` to avoid namespace collision with sklearn's PCA.
- Replaced project-internal logging with stdlib `logging` where applicable.
- Used direct SVD on centered data instead of accumulator-based covariance to keep the MVP path obviously correct.
- `quantize.py`: trimmed `LinearLayer` to quantize/dequantize round-trip; constant-row branch added so all-equal rows round-trip exactly.
- `knn.py`: stripped `ChunkedKNN` to its inner GEMM+topk step; tiling layer is now a separate concern in `pipeline/tiling.py`. Caller is responsible for normalisation (clearer contract than mixing it in here).
- `master_to_dict.py`: Path B chosen for MVP. pcadi's real GPU projection (`MasterPattern.interpolate` + `EBSDGeometry.get_coords_sample_frame` + `qu_apply` quaternion rotation of detector grid + `rosca_lambert_side_by_side` Lambert square mapping) requires extracting `bruker_geometry_to_SE3`, `se3_exp_map_om`, Laue-group mapping from kikuchipy phase metadata, plus careful PC-convention translation between Bruker (kikuchipy) and the SE3-encoded pcadi geometry. Conservatively >200 LOC of vendoring. Wrapping kikuchipy unblocks the indexer (Phase 3) without committing to a specific projection chain; the swap is API-stable (see "Known follow-ups" below).

## Known follow-ups

- `master_to_dict.py` currently wraps kikuchipy's CPU forward-projection.
  Replace with pcadi's GPU-side `MasterPattern` + `EBSDGeometry`
  projection (utils.py 3321–4649) for true end-to-end GPU dict
  generation. Public API is stable; this swap is a drop-in. Concretely
  this needs:
  1. Vendor `bruker_geometry_to_SE3`, `se3_exp_map_om`,
     `rosca_lambert_side_by_side`, `qu_apply`, plus enough of
     `EBSDGeometry` to expose `get_coords_sample_frame()`.
  2. Vendor a slim `MasterPattern.interpolate` (~50 LOC,
     `grid_sample` over Lambert square).
  3. Adapter that pulls the (NH, SH) hemispheres out of
     `kp.signals.EBSDMasterPattern.data` and resolves the Laue group
     from the phase / point group.
  4. Convert `orix.quaternion.Rotation` to pcadi's quaternion layout
     (it already follows the (w, x, y, z) convention used by orix).

## License

MIT License

Copyright (c) 2025 ZacharyVarley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
