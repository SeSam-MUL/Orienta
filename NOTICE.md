# Third-Party Notices

Orienta is licensed under the **GNU General Public License v3.0 or later**
(see [LICENSE](LICENSE)). It builds on, and in places contains code derived from,
the third-party works listed below. Each retains its own license.

## Vendored / derived source code

| Component | Location | Origin | License |
|-----------|----------|--------|---------|
| pcadi (pattern-center–aware dictionary indexing core) | `backend/dict_gpu/_pcadi/` | Vendored from the `pcadi` project | MIT |
| GPU dictionary / spherical projection math | `backend/dict_gpu/`, `backend/spherical_gpu/` | Translated/adapted from **kikuchipy** 0.11.3 (Lambert & detector projection) | GPL-3.0-or-later |
| Spherical-harmonic math kernels (Wigner-d, SHT, cubochoric grids) | `backend/spherical_gpu/_math/` | Vendored/adapted from **ebsdtorch** (Zachary Varley, 2024); see `LICENSE.ebsdtorch` there | MIT |
| GPU dictionary projection (second package) | `backend/dictionary_gpu/` | Vectorised port of **kikuchipy**'s direction-cosine / Lambert projection (itself adapted from EMsoft) | GPL-3.0-or-later |
| GPU forward model (Monte-Carlo, master pattern, SHT writer) | `backend/forward_sim/` | Ported from **ebsdtorch** (MIT); plus tabulated data transcribed from **EMsoft** and **SHTfile** — see the two rows below | MIT / BSD-3-Clause |
| WEKO elastic scattering coefficients (Z = 1..98) | `backend/forward_sim/crystal/scattering_factors.py` | Transcribed verbatim from **EMsoft** `Source/EMsoftLib/others.f90` (GETWK tables) | BSD-3-Clause — `licenses/EMsoft-License.txt` |
| Monte-Carlo RNG stream (LFSR113) | `backend/forward_sim/mc/cupy_mc_kernel.py` | Ported from **EMsoft** `EMMC.cl` | BSD-3-Clause — `licenses/EMsoft-License.txt` |
| Two 230-entry space-group lookup tables | `backend/forward_sim/io/sht_writer.py` | Ported from **SHTfile** `sht_file.in.hpp` (EMsoft-org/SHTfile) | BSD-3-Clause — `licenses/SHTfile-License.txt` |
| Oxford H5OINA reader workaround | `safe_loader.py` | Body copied verbatim from **kikuchipy** 0.11.3 `oxford_h5ebsd/_api.py`, with a widened `except` clause | GPL-3.0-or-later |

Because parts of this project are derived from GPL-3.0 code (kikuchipy), the
combined work is distributed under GPL-3.0-or-later. The MIT-licensed pcadi code
is compatible with that combination; its copyright notice is retained in the
source files under `backend/dict_gpu/_pcadi/`.

## Algorithm / method attributions

Orienta implements published EBSD methods. Where it provides an **independent
reimplementation** of a method, the original authors are credited here and should
be cited in any publication that uses these features:

- **Spherical-harmonic-transform (SHT) indexing** — `backend/spherical_gpu/` is an
  independent GPU reimplementation of the spherical-indexing method of
  **W. C. Lenthe, S. Singh & M. De Graef, "A spherical harmonic transform approach
  to the indexing of electron back-scattered diffraction patterns," _Ultramicroscopy_
  207, 112841 (2019)**, as realized in the **EMSphInx** project (EMsoft-org/EMSphInx,
  Carnegie Mellon University). Orienta contains **no EMSphInx source code**; the
  implementation was derived from the published method and validated independently.
- **Dictionary indexing** — follows the EMsoft / kikuchipy dictionary-indexing
  approach (S. Singh & M. De Graef; the kikuchipy project).
- **Hough/Radon indexing** — provided via **PyEBSDIndex** (U.S. NRL; see below).
- **Monte-Carlo electron scattering + dynamical master patterns** — follow the
  **EMsoft** methods (M. De Graef et al.).

These academic attributions are in addition to the source-license obligations above.

## Key runtime dependencies

Except for the two bundled JavaScript files noted below, these are required
dependencies **installed separately** and not redistributed in this repository.
Each is governed by its own license:

- **kikuchipy**, **orix**, **diffsims**, **hyperspy** — GPL-3.0 (these GPL-3.0
  upstreams are the reason the combined work is licensed GPL-3.0-or-later)
- **PyQt5** (Riverbank Computing) — **GPL-3.0**. A leftover of the retired Qt
  interface; it is still listed in `requirements.txt` but no shipped code path
  imports it. Note that Riverbank licenses PyQt5 under GPL v3 **only**, not
  "or later" — removing this dependency is tracked in the README's
  *Known limitations*.
- **NumPy**, **SciPy**, **pandas**, **scikit-learn**, **scikit-image**, **matplotlib**, **h5py**, **Pillow** — BSD / PSF-style
- **PyTorch**, **torchvision** — BSD-3-Clause
- **FastAPI**, **uvicorn**, **pydantic**, **httpx**, **requests** — MIT / BSD
- **React**, **Vite**, **Electron**, **zustand** — MIT

**Redistributed in this repository** (both under `frontend/public/vendor/`, loaded
directly by `frontend/index.html`, dashboard background only):

- **three.js** — MIT. Its `@license` header (Copyright 2010-2021 Three.js Authors)
  is intact at the top of `three.min.js`.
- **vanta.js** 0.5.24 (Copyright 2020 Teng Bao) — MIT. The minified bundle carries
  no header of its own, so the required notice is reproduced in
  `frontend/public/vendor/LICENSE.vanta.txt`.

## Public-domain / government works

- **PyEBSDIndex** (Hough/Radon EBSD indexing) — developed by the **U.S. Naval
  Research Laboratory (NRL)**; as a work of U.S. Government employees it is **in
  the public domain** (17 U.S.C. §105) under a custom permissive grant. We
  acknowledge the NRL as the original source and note any modifications, per its
  terms. Public-domain works are compatible with GPL-3.0.
  Source: https://github.com/USNavalResearchLaboratory/PyEBSDIndex

## Optional proprietary components (NOT redistributed)

- **NVIDIA CUDA runtime** (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`,
  `cupy-cuda12x`, etc.) — proprietary NVIDIA libraries, **optional** (only for GPU
  acceleration). Users install these themselves via pip; this project does **not**
  bundle or redistribute them (the GPL "System Library" exception applies).

## External tools (optional, not bundled)

- **EMsoft** / **EMSphInx** — used via WSL for Monte-Carlo / master-pattern
  simulation and spherical indexing. Installed and licensed separately by the user.

If you redistribute this software, retain this NOTICE file and the LICENSE file,
and preserve all copyright and license headers in the vendored source.
