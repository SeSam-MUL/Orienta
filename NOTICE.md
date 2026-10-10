# Third-Party Notices

Orienta's own source code is licensed under the **GNU General Public License
v3.0 or later** (see [LICENSE](LICENSE)). It builds on, and in places contains
code derived from, the third-party works listed below. Each retains its own
license.

**License of the distributed whole.** Because the shipped environment currently
includes **PyQt5**, which Riverbank Computing licenses under GPL v3 *only* (see
*Key runtime dependencies*), a binary or installer that contains PyQt5 can be
conveyed only under **GPL-3.0** (the "or later" option does not apply to that
combination). Orienta's own files keep their GPL-3.0-or-later grant; the
restriction disappears once PyQt5 is removed from the dependency set.

## Vendored / derived source code

| Component | Location | Origin | License |
|-----------|----------|--------|---------|
| pcadi (pattern-center–aware dictionary indexing core) | `backend/dict_gpu/_pcadi/` | Vendored from the `pcadi` project (Zachary Varley, 2025); see `__SOURCE.md` there | MIT |
| GPU dictionary / spherical projection math | `backend/dict_gpu/`, `backend/spherical_gpu/` | Translated/adapted from **kikuchipy** 0.11.3 (Lambert & detector projection) | GPL-3.0-or-later |
| Spherical-harmonic math kernels (Wigner-d, SHT, cubochoric grids) | `backend/spherical_gpu/_math/` | Vendored/adapted from **ebsdtorch** (Zachary Varley, 2024); see `LICENSE.ebsdtorch` and `__SOURCE.md` there | MIT |
| Weighted DCT/DST linear layers | `backend/spherical_gpu/_math/sht.py` (`weightedDCST`) | Adapted from **torch-dct** (Ziyang Hu, 2018) | MIT — `licenses/torch-dct-License.txt` |
| GPU dictionary projection (second package) | `backend/dictionary_gpu/` | Vectorised port of **kikuchipy**'s direction-cosine / Lambert projection (itself adapted from EMsoft) | GPL-3.0-or-later |
| GPU forward model (Monte-Carlo, master pattern, SHT writer) | `backend/forward_sim/` | Ported from **ebsdtorch** (MIT); plus tabulated data transcribed from **EMsoft** and **SHTfile** — see the rows below | MIT / BSD-3-Clause |
| WEKO elastic scattering coefficients (Z = 1..98) and phonon form factor | `backend/forward_sim/crystal/scattering_factors.py` | Transcribed verbatim from **EMsoft** `Source/EMsoftLib/others.f90` (GETWK tables, FPHON). EMsoft's file header states this block is "routines written by other people, included with permission" — the Weickenmeier–Kohl implementation, included with permission from H. Kohl (A. Weickenmeier & H. Kohl, *Acta Cryst.* A47, 590, 1991). The numbers are fitted physical constants. | BSD-3-Clause — `licenses/EMsoft-License.txt` |
| Monte-Carlo RNG stream (LFSR113) | `backend/forward_sim/mc/cupy_mc_kernel.py` | Ported from **EMsoft** `EMMC.cl` | BSD-3-Clause — `licenses/EMsoft-License.txt` |
| Two 230-entry space-group lookup tables, CRC-32C table, harmonic packing (`NumHarm`/`PackHarm`) | `backend/forward_sim/io/sht_writer.py` | Ported from **SHTfile** `sht_file.in.hpp` (EMsoft-org/SHTfile, W. C. Lenthe, 2019) | BSD-3-Clause — `licenses/SHTfile-License.txt` |
| Harmonic unpacking (`NumHarm`/`UnpackHarm`) of the `.sht` reader | `backend/spherical_gpu/pipeline/sht_io.py` | Ported from **SHTfile** `sht_file.in.hpp` | BSD-3-Clause — `licenses/SHTfile-License.txt` |
| Oxford H5OINA reader workaround | `safe_loader.py` | Body copied verbatim from **kikuchipy** 0.11.3 `oxford_h5ebsd/_api.py`, with a widened `except` clause | GPL-3.0-or-later |
| Ten toolbar icons (`alert-circle`, `archive`, `check-circle`, `edit-3`, `file`, `file-plus`, `files`, `folder`, `refresh-cw`, `save`) | `crystal-structures-for-ebsd-main/calculationxtal/icons/*.svg` | **Feather Icons** (Cole Bemis) | MIT — `licenses/Feather-License.txt` |

Because parts of this project are derived from GPL-3.0 code (kikuchipy), the
combined work is distributed under the GPL (see the note on the distributed
whole above). The MIT- and BSD-licensed components are compatible with that
combination; their copyright notices are retained in the source files and in
`licenses/`.

`crystal-structures-for-ebsd-main/` (the CIF → `.xtal` converter, the
Debye–Waller table and the EMsoft automation module) is first-party code of the
Orienta authors and is covered by the repository's LICENSE; the `LICENSE` file
inside that directory says so.

## Reference data

- **Debye–Waller factors** — `crystal-structures-for-ebsd-main/calculationxtal/DWF.xlsx`
  holds 39 room-temperature B values (one per element), 38 of them transcribed
  from **L.-M. Peng, G. Ren, S. L. Dudarev & M. J. Whelan, "Debye–Waller Factors
  and Absorptive Scattering Factors of Elemental Crystals", *Acta Cryst.* A52,
  456–470 (1996), Table 1** (IUCr), and one (Mn) from a published plot; each row
  carries its reference, see `DWF_SOURCES.md` next to it. These are measured
  physical constants, not the article. The article and its supplement are not
  redistributed.
- **Crystal structures** — no CIF, `.xtal`, `.sht` or master-pattern file from a
  database with redistribution limits (ICSD, SpringerMaterials/Pauling File) is
  part of this repository or of any release package; the build refuses such
  files. `sample_data/phases/` contains only Crystallography Open Database
  (CC0) structures, see its `PROVENANCE.md`.

## Algorithm / method attributions

Orienta implements published EBSD methods. Where it provides an **independent
reimplementation** of a method, the original authors are credited here and should
be cited in any publication that uses these features:

- **Spherical-harmonic-transform (SHT) indexing** — `backend/spherical_gpu/` is an
  independent GPU reimplementation of the spherical-indexing method of
  **W. C. Lenthe, S. Singh & M. De Graef, "A spherical harmonic transform approach
  to the indexing of electron back-scattered diffraction patterns," _Ultramicroscopy_
  207, 112841 (2019)**, as realized in the **EMSphInx** project (EMsoft-org/EMSphInx,
  Carnegie Mellon University), and of the spherical cross-correlation of
  **R. Hielscher, F. Bartel & T. B. Britton, _Ultramicroscopy_ 207, 112836 (2019)**.
  Orienta contains **no EMSphInx program code**; the correlation, decoding and
  refinement were written from the published method (on ebsdtorch's MIT
  primitives) and validated independently. The only material taken from the
  EMsoft-org projects is listed in the table above: BSD-licensed lookup tables
  and packing routines of the open `.sht` file format (SHTfile), and EMsoft
  data tables. An independent reimplementation addresses copyright only; see
  the patent note under *External tools*.
- **Dictionary indexing** — follows the EMsoft / kikuchipy dictionary-indexing
  approach (S. Singh & M. De Graef; the kikuchipy project).
- **Hough/Radon indexing** — the Radon-transform band detection and the band
  indexing (triplet voting) are performed by **PyEBSDIndex** (U.S. NRL; see
  below). Cite **D. J. Rowenhorst, P. G. Callahan & H. W. Ånes, "Fast Radon
  transforms for high-precision EBSD orientation determination using
  PyEBSDIndex," _J. Appl. Cryst._ 57(1), 3–19 (2024),
  doi:10.1107/S1600576723010221**, together with kikuchipy wherever a
  Hough-indexed result is reported.
- **Monte-Carlo electron scattering + dynamical master patterns** — follow the
  **EMsoft** methods (M. De Graef et al.).

These academic attributions are in addition to the source-license obligations above.

> **Machine-readable form.** The same attributions are kept as CSL-JSON in
> [`backend/api/services/citations/library.json`](backend/api/services/citations/library.json),
> mapped to the pipeline steps that use them in
> [`backend/api/services/citations/steps.py`](backend/api/services/citations/steps.py).
> That is what the app's "Citations for this result" panel and the `/Citations`
> group of an exported `.h5` are rendered from; this file stays the
> human-readable licence document.

## Key runtime dependencies

Except for the two bundled JavaScript files noted below, these are required
dependencies **installed separately** (by `pip` from PyPI and the PyTorch
index on Windows and Linux; by micromamba from conda-forge on macOS, see
below) and not redistributed in this repository. Each is governed by its own
license:

- **kikuchipy**, **orix**, **diffsims**, **hyperspy**, **rosettasciio** —
  GPL-3.0-or-later (these GPL-3.0 upstreams are the reason the combined work is
  licensed under the GPL)
- **PyQt5** (Riverbank Computing) — **GPL-3.0 only**. A leftover of the retired
  Qt interface. It is pinned in both lock files and **is imported** by the
  simulation helpers `simulation/crystal_sync.py`, `database_browser.py`,
  `file_sync_manager.py`, `server_mode_manager.py` and `simulation_worker.py`
  (reachable from the simulation API), and by the legacy converter window
  `crystal-structures-for-ebsd-main/calculationxtal/Xtal_Generator_GUI.py`,
  which ships but is not started by the app. Riverbank licenses PyQt5 under GPL v3
  **only**, not "or later"; see the note on the distributed whole at the top.
  When installed by `pip`, PyQt5 pulls in **PyQt5-Qt5** (the Qt 5 runtime
  libraries, LGPL-3.0; Riverbank notes that a few Qt modules such as QtCharts
  are GPL-3.0 instead). Removing PyQt5 from the dependency set is planned.
- **PyEBSDIndex** — public domain, see below.
- **NumPy**, **SciPy**, **pandas**, **scikit-learn**, **scikit-image**,
  **numba**, **llvmlite**, **dask**, **h5py**, **spglib**, **diffpy.structure**,
  **psutil**, **starlette**, **uvicorn**, **websockets**, **httpx** — BSD
- **matplotlib** — matplotlib (PSF-style) license; **Pillow** — MIT-CMU (HPND)
- **PyTorch**, **torchvision** — BSD-3-Clause (their wheels bundle further
  components, see `torch`'s own LICENSE and NOTICE files)
- **FastAPI**, **pydantic**, **pymatgen**, **faiss-cpu**, **openpyxl**,
  **pyyaml**, **beautifulsoup4**, **cupy** — MIT
- **requests**, **watchdog**, **ray** — Apache-2.0
- **tqdm** — MPL-2.0 and MIT; **certifi** — MPL-2.0
- **React**, **Vite**, **Electron**, **zustand**, **three.js** — MIT;
  **lucide-react** — ISC

**The bundled user interface.** `frontend/dist/` is a build that inlines about
three hundred npm packages (React, plotly.js, mapbox-gl, axios, i18next,
react-router, react-window and their dependencies; MIT, ISC, BSD and similar
permissive licenses). Their copyright and permission notices do not survive
minification, so the build writes them to
`frontend/dist/THIRD-PARTY-LICENSES.txt` (generated by
`frontend/scripts/third_party_licenses.mjs` from the lock file and the
packages' own LICENSE files). That file ships with every release.

**The Windows and Linux packages' Python.** The installer downloads a standalone
CPython from python-build-standalone (Astral). The interpreter archive bundles
OpenSSL 3 (Apache-2.0), SQLite (public domain), libffi, Expat and mpdecimal
(MIT/BSD-style), liblzma, zlib, bzip2, Tcl/Tk/Tix, and, on Windows, Microsoft's
Visual C++ runtime DLLs. The open-source components' license texts are
reproduced in `licenses/python-build-standalone/`. The Microsoft runtime DLLs
are Microsoft "Distributable Code" under the Visual Studio license terms; the
conditions Microsoft attaches to passing them on are printed in the
`LICENSE.txt` inside the downloaded interpreter archive, which the installer
keeps next to the interpreter (`python.exe` on Windows, `bin/python3` on
Linux).

**The macOS package's Python and packages.** On macOS the installer does not
use pip. It downloads **micromamba** (mamba-org, BSD-3-Clause; version and
SHA-256 pinned in `electron/platform.js`, fetched from the
`mamba-org/micromamba-releases` GitHub releases) onto the user's machine, runs
it once to build the Python environment from the lock file
`orienta-macos-lock.yml`, and keeps the binary in Orienta's runtime folder for
later repairs. Orienta does not redistribute micromamba; the person installing
Orienta obtains it from mamba-org. Because it stays installed as part of
Orienta's runtime, its license is reproduced in
`licenses/micromamba-License.txt` all the same. The downloaded asset is a bare
executable; it statically links further open-source libraries (libsolv,
libcurl, OpenSSL, libarchive, zstd, yaml-cpp, reproc, fmt, spdlog,
nlohmann-json, simdjson and others, BSD/MIT/OpenSSL-style), whose license
texts are in the conda-forge package the binary is extracted from, as that
file explains.

The environment micromamba builds comes entirely from the **conda-forge**
channel on `conda.anaconda.org` (`environment-macos.yml` declares
`nodefaults`; the lock file names no other channel, and the installer runs
micromamba with `--no-rc --no-env`, so no channel configured on the user's
machine can be added). Anaconda's own `defaults` repository
(`repo.anaconda.com`, the one under Anaconda's paid terms) is never used. The
packages are the conda-forge builds of the same projects listed above, plus
`llvm-openmp` (Apache-2.0 with LLVM exception) and Apple's Accelerate BLAS
(part of macOS). They are installed on the user's machine by micromamba and
not redistributed by Orienta. Every conda-forge package carries its own
license files in its conda archive (`info/licenses/`); the installer removes
the downloaded archives after building the environment, so those copies are
not kept on the user's machine, and a user who redistributes the environment
must take them from the conda-forge channel. The lock includes the conda-forge
`pyqt` 5.15 build, so the note on PyQt5 (GPL-3.0 only) applies to the macOS
package exactly as to the others.

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
  the public domain** (17 U.S.C. §105) under a custom permissive grant whose
  conditions are: acknowledge the NRL as the original source, and mark
  derivative or modified versions as such. Orienta uses the unmodified package
  from PyPI; the only runtime adaptation is a wrapper around its triplet
  library builder in `ebsd_utils.py`, which is documented there. Public-domain
  works are compatible with GPL-3.0. The NRL also states that it would
  appreciate acknowledgment when the software is used; the academic reference
  is given under *Algorithm / method attributions* above and in the "How to
  cite" section of the README.
  Source: https://github.com/USNavalResearchLaboratory/PyEBSDIndex

## Optional proprietary components (NOT redistributed)

- **NVIDIA CUDA runtime and cuDNN** (`nvidia-*-cu12` wheels, `cupy-cuda12x`, and
  the CUDA/cuDNN libraries bundled inside the `torch` CUDA wheel) — proprietary
  NVIDIA libraries under NVIDIA's own EULAs, **optional** (GPU acceleration
  only). They are named in `requirements-lock-gpu.txt` so that the installer
  can fetch them from PyPI and the PyTorch index onto the user's machine when
  the user chooses the GPU build; this repository and the release packages do
  **not** contain them. Anyone who assembles and passes on a pre-populated GPU
  environment (an offline installer, a lab image) redistributes them and must
  check NVIDIA's terms themselves.

## External tools (optional, not bundled)

- **EMsoft** (BSD-3-Clause, Marc De Graef Research Group / Carnegie Mellon
  University) and **EMSphInx** (GPL-2.0-or-later, De Graef Group / CMU; a
  commercial license is offered by CMU's Center for Technology Transfer and
  Enterprise Creation) — optional external programs for Monte-Carlo and
  master-pattern simulation and for spherical indexing. Orienta bundles
  **neither**: the in-app installer clones them from their upstream
  repositories and builds them inside WSL on the user's own machine. Orienta
  starts the resulting binaries as **separate processes** and exchanges data
  through files, so they are separate programs, not part of Orienta.
  Both licenses permit commercial use. EMsoft's `License.txt` contains a
  sentence saying that "for EMSphInx, a non-commercial license applies, the
  details of which can be found in the Source/EMSphInx/license.txt file";
  that file does not exist in EMsoft, and EMSphInx has been distributed under
  GPL-2.0-or-later since its first public release (EMsoft's own ReadMe:
  "EMSphInx is distributed under a GPL license"). The GPL constrains the terms
  of *distribution*, not commercial *use*.
- **Patent note.** The EMSphInx ReadMe states that "the central indexing
  algorithm is covered by a provisional patent application" (unchanged since
  October 2019). As of 2026-09-23 no published or granted patent on that method
  could be found in the sources available to us; that is "not found", not
  "does not exist". The GPL-2.0 text grants no express patent rights; GPL-3.0
  §11 does, and EMSphInx's "or later" grant lets a recipient take EMSphInx
  under GPL-3.0. That grant travels with EMSphInx's code only; it does **not**
  extend to Orienta's own independent implementation, which contains none of
  it. An independent reimplementation is a defence against copyright claims,
  not against a patent. Users who commercialise spherical indexing should take
  their own advice.

## MATLAB reference scripts (`matlab_testskripts/`)

These scripts are reference material, not part of the running application: they
document the EBSD post-processing workflow that the Python module in `analysis/`
was ported from.

**Authorship and license.** The scripts were written by Dr. Irmgard
Weißensteiner; the copyright is held by her and Montanuniversität Leoben, who
have agreed to their publication here under the GNU General Public License,
version 2 or (at your option) any later version
(`SPDX-License-Identifier: GPL-2.0-or-later`). Each of her files
carries a header naming her and the license; the full text and the copyright
line are in the LICENSE file of that folder
<!-- Cited as markdown links on purpose: scripts/build_runtime_package.py
     treats backticked LICENSE paths as files the installed package must
     contain, and matlab_testskripts/ is deliberately not packaged. -->
([matlab_testskripts/LICENSE](matlab_testskripts/LICENSE)), and
[matlab_testskripts/README.md](matlab_testskripts/README.md) explains what the
scripts are. The copy here is the state Orienta was ported from; the author has
developed the scripts further since. GPL-2.0-or-later is compatible with
Orienta's GPL-3.0: the scripts may be used under version 3 alongside the rest of
this repository. Two third-party attributions apply to them in addition.

**freezeColors (bundled).** `matlab_testskripts/freezeColors.m` (v2.3) is a
third-party MATLAB utility by John Iversen, not written by this project. Its own
header carries the author's grant, and the condition attached to it:

> Free for all uses, but please retain the following:
>   Original Author:
>   John Iversen, 2005-10
>   john_iversen@post.harvard.edu

That block must stay in the file. The upstream project
(<https://github.com/jiversen/freezeColors>) added a formal MIT licence in 2024,
after this revision; its text is in `licenses/freezeColors-Upstream-MIT.txt`,
named so that nobody reads it as the grant our copy shipped under. Under either reading the terms are permissive, require attribution
only, and are compatible with the GNU GPL.

**MTEX (referenced, not redistributed).** The scripts call functions of the MTEX
toolbox (for example `EBSD`, `calcGrains`, `calcDensity`, `crystalSymmetry`,
`orientation`, `grain2d`, `degree`). **No MTEX source code is included in, or
redistributed with, this repository.** MTEX is a separate work that the user
installs independently, and the scripts cannot run without it. MTEX is free
software distributed under the GNU General Public License, version 2; see the
`COPYING.txt` that MTEX itself ships for the licence text it carries. Because no
MTEX code is distributed here, and the scripts cannot run until the user installs
MTEX themselves, no combined work arises and no licence-compatibility question
arises with it.

- Project website: <https://mtex-toolbox.github.io/>
- Source repository: <https://github.com/mtex-toolbox/mtex>

MTEX asks users to cite whichever paper best fits the application and keeps the
list at <https://mtex-toolbox.github.io/publications>. The foundational ones:

> F. Bachmann, R. Hielscher, H. Schaeben: *Texture Analysis with MTEX - Free and
> Open Source Software Toolbox.* Solid State Phenomena **160** (2010), 63-68.
> DOI: 10.4028/www.scientific.net/SSP.160.63

> R. Hielscher, H. Schaeben: *A novel pole figure inversion method: specification
> of the MTEX algorithm.* Journal of Applied Crystallography **41**(6) (2008),
> 1024-1037. DOI: 10.1107/S0021889808030112

## Where the license texts are

- `LICENSE` — GPL-3.0, the license of Orienta.
- [matlab_testskripts/LICENSE](matlab_testskripts/LICENSE) — GPL-2.0-or-later, the
  MATLAB reference scripts (not part of the installed package; a markdown link,
  not a backticked path, for the same reason as above).
- `licenses/` — EMsoft, SHTfile, torch-dct, freezeColors (upstream MIT, for reference), Feather Icons, micromamba (macOS),
  and the components of the standalone Python (`licenses/python-build-standalone/`).
- `backend/spherical_gpu/_math/LICENSE.ebsdtorch`, `backend/dict_gpu/_pcadi/__SOURCE.md`
  — the vendored MIT code.
- `frontend/public/vendor/LICENSE.vanta.txt` — the two bundled JavaScript files.
- `frontend/dist/THIRD-PARTY-LICENSES.txt` — the npm packages inlined into the
  built user interface (generated at build time).

If you redistribute this software, retain this NOTICE file, the LICENSE file
and the `licenses/` directory, and preserve all copyright and license headers
in the vendored source.
