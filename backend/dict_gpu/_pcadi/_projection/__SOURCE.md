# Translated from kikuchipy._master_pattern (Path A)

**kikuchipy version pinned:** 0.11.3 (see requirements.txt)
**Reference file:** `kikuchipy/signals/util/_master_pattern.py`
**Imported on:** 2026-05-11
**Translation guide:** `tasks/kikuchipy_projection_reference.md`

## Function map

| File here | Reference symbol | Reference lines | Modifications |
|---|---|---|---|
| `direction_cosines.py::compute_direction_cosines` | `_get_direction_cosines_for_fixed_pc` | 149-235 | vectorised torch; returns (h, w, 3) not flat (h*w, 3); FP32/FP16 dtype param |
| `lambert.py::vector_to_lambert` | `_vector2lambert` | 578-615 | vectorised torch with torch.where for the \|x\|>=\|y\| branch |
| `lambert.py::lambert_to_vector` | `_lambert2vector` | 764-807 | vectorised torch likewise |
| `sample.py::sample_master` | `_get_lambert_interpolation_parameters` + `_get_pixel_from_master_pattern` | 627-756 | torch.nn.functional.grid_sample replaces hand-rolled bilinear (align_corners=True, padding_mode=border); hemisphere dispatch via z>=0 mask |
| `project.py::_rotate_directions` | `kikuchipy._utils.numba.rotate_vector` | numba.py 62-81 | vectorised active rotation R(q)=qvq* expanded to a 3x3 matrix per quaternion, applied via einsum across (n, det_h, det_w, 3) |
| `project.py::project_master_to_detector` | `_project_single_pattern_from_master_pattern` (+ batched wrapper) | 496-575, 346-415 | composes rotate + lambert + sample; vectorised over rotations in one pass; uses `q` directly (NOT q^-1) — kikuchipy's `rotate_vector(rotation, dc)` applies q itself |

(later tasks add rows here)

## License notes

kikuchipy is GPLv3. The math we translate is from published EBSD references
(Rosca-Lambert projection, standard direction-cosine geometry); the
*implementation* in PyTorch is independent. No code is copied verbatim.
