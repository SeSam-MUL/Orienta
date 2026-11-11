# Vendored from ZacharyVarley/pcadi

- **Commit:** 88f676e6f44812900b3c873692aa1a1c9e48b746
- **License:** MIT (see verbatim text below)
- **Imported on:** 2026-05-10

## Extracted symbols

| File here | Source file | Source symbol(s) | Source lines (approx) | Modifications |
|---|---|---|---|---|
| `pca.py` | `utils.py` | inspired by `OnlineCovMatrix` | 2080–2400 | Replaced streaming accumulator with direct SVD on centered data; renamed `PCA` → `GpuPCA`; removed CPU/NumPy fall-back path; added explicit shape & component-count validation |
| `quantize.py` | `utils.py` (also `utils_knn.py`) | inspired by `LinearLayer` quantization | utils.py 4650–4720, utils_knn.py 8–110 | Trimmed to quantize/dequantize only (dropped quantised linear-forward path); per-row scale + zero-point with constant-row safeguard |
| `knn.py` | `utils.py` (also `utils_knn.py`) | inspired by `ChunkedKNN` GEMM core | utils.py 4720–5651, utils_knn.py 110–298 | Stripped to single-pass GEMM + top-k; chunking moved to pipeline/tiling.py; explicit FP16/FP32 toggle |
| `master_to_dict.py` | (none — composed here) | n/a | n/a | Generates the dictionary on the GPU with the forward projection in `_projection/` (rotate the detector directions, map them onto the square Lambert grid, sample the master hemispheres bilinearly). That projection is translated from kikuchipy, not taken from pcadi; see `_projection/__SOURCE.md`. A CPU fallback that calls `master_pattern.get_patterns(...)` and uploads the result stays available through `_use_path_b=True`. |

## Modifications from upstream

- Removed CPU/NumPy fall-back path; this module is GPU-by-construction (CPU torch tensors also work but are not optimised).
- Renamed `PCA` → `GpuPCA` to avoid namespace collision with sklearn's PCA.
- Replaced project-internal logging with stdlib `logging` where applicable.
- Used direct SVD on centered data instead of accumulator-based covariance to keep the implementation simple and its result easy to verify.
- `quantize.py`: trimmed `LinearLayer` to quantize/dequantize round-trip; constant-row branch added so all-equal rows round-trip exactly.
- `knn.py`: stripped `ChunkedKNN` to its inner GEMM+topk step; tiling layer is now a separate concern in `pipeline/tiling.py`. Caller is responsible for normalisation (clearer contract than mixing it in here).
- `master_to_dict.py`: pcadi's own GPU projection (`MasterPattern.interpolate` + `EBSDGeometry.get_coords_sample_frame` + `qu_apply` quaternion rotation of the detector grid + `rosca_lambert_side_by_side` Lambert square mapping) is not vendored: it needs `bruker_geometry_to_SE3`, `se3_exp_map_om`, a Laue-group mapping from the kikuchipy phase metadata and a translation between the Bruker (kikuchipy) and the SE3-encoded pcadi detector-geometry conventions. The projection used instead is the independent PyTorch translation of kikuchipy's reference math in `_projection/`, which follows the kikuchipy conventions directly.

## Projection used by `master_to_dict.py`

`gpu_master_to_dict` runs the GPU projection in `_projection/` by default. The
CPU fallback (`_use_path_b=True`) wraps kikuchipy's `get_patterns` and uploads
the resulting dictionary block to the GPU; it is kept only as an emergency
fallback and as a reference the GPU result is compared against.

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
