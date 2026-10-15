<p align="center"><img src="branding/orienta-icon-512.png" width="116" alt="Orienta logo"></p>

# Orienta — EBSD Pattern Analysis

> **Making EBSD indexing accessible**

[![Latest release](https://img.shields.io/github/v/release/SeSam-MUL/Orienta?label=release)](https://github.com/SeSam-MUL/Orienta/releases/latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22664080.svg)](https://doi.org/10.5281/zenodo.22664080)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
![Platforms: Windows, macOS (Apple Silicon), Linux](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%28Apple%20Silicon%29%20%7C%20Linux-lightgrey)

A desktop application for **EBSD (Electron Backscatter Diffraction)** pattern
analysis: load Oxford H5OINA / EDAX H5 data, refine the pattern center, index
crystal phases (Hough / Dictionary / Spherical), integrate EDS chemistry, and
produce phase maps, grain/texture analyses, and publication figures.

Built on the [kikuchipy](https://kikuchipy.org/), [orix](https://orix.readthedocs.io/)
and [diffsims](https://diffsims.readthedocs.io/) scientific stack. Hough indexing
(the Radon-transform band detection and the band indexing) is performed by
[PyEBSDIndex](https://github.com/USNavalResearchLaboratory/PyEBSDIndex), see
[How to cite](#how-to-cite).

> **Status: beta (v0.4.7).** Everything listed under *Features* is reachable in the
> app, but some of it is newer than its test coverage, and a few pieces are less
> complete than their name suggests. Read
> [Known limitations](#known-limitations--to-be-done) before you rely on a result.

## Download

<p>
  <a href="https://github.com/SeSam-MUL/Orienta/releases/latest/download/Orienta-Windows-x64-Setup.exe"><img alt="Download for Windows (64-bit)" src="https://img.shields.io/badge/Download-Windows%20%2864--bit%29-2ea44f?style=for-the-badge"></a>
  <a href="https://github.com/SeSam-MUL/Orienta/releases/latest/download/Orienta-macOS-arm64.dmg"><img alt="Download for macOS (Apple Silicon)" src="https://img.shields.io/badge/Download-macOS%20%28Apple%20Silicon%29-2ea44f?style=for-the-badge"></a>
  <a href="https://github.com/SeSam-MUL/Orienta/releases/latest/download/Orienta-Linux-x86_64.AppImage"><img alt="Download for Linux (x86-64)" src="https://img.shields.io/badge/Download-Linux%20%28x86--64%29-2ea44f?style=for-the-badge"></a>
</p>

Each button downloads the newest version. No Python and no programming needed:
each package sets up its own Python on first start. The macOS package needs
macOS 14 or newer on Apple Silicon (M1 and later); Macs with an Intel processor are
not supported. Checksums, release notes and older versions are on the
[releases page](https://github.com/SeSam-MUL/Orienta/releases).

None of the packages is signed with a paid certificate, so Windows and macOS ask
once whether you really want to open it. The first start downloads Python and the
scientific packages, so it needs an internet connection and several GB of free disk
space. [INSTALL.md](INSTALL.md#the-ready-made-packages-the-short-way) shows what to
click, how much space each system needs, and how to check the `.sha256` file that
sits beside each download.

## Features

- Load & visualize EBSD patterns (Oxford H5OINA, EDAX H5 on square grids, EDAX H5
  on hexagonal grids resampled to a square grid, EDAX UP1/UP2 + `.osc`; lazy load
  for large files)
- Pattern Center refinement (incl. pixel-wise drift correction)
- Indexing: Hough, Dictionary (CPU + GPU), Spherical (built-in GPU indexer —
  the default — or EMSphInx via WSL)
- EDS integration: element maps, Counts → Wt% → At%, chemistry-based phase suggestion
- Layered phase maps (IPF / BC / CI / EDS overlays, configurable scalebar, export)
- EBSD analysis: grain reconstruction, KAM/GOS, texture components (14, FCC/BCC
  presets) with texture index & entropy, recrystallization, Excel export
- Crystal database: CIF → .xtal conversion, Debye–Waller editor
- Simulation: EMsoft Monte-Carlo + master pattern + SHT, or an EMsoft-free GPU forward model
- HDF5 explorer (patterns, EDS maps, electron images)

## Documentation

- **Installation:** [INSTALL.md](INSTALL.md) — beginner's step-by-step setup guide.
- **User guides:** [docs/user-guide/](docs/user-guide/) — one page per app feature
  (EBSD Viewer, Indexing, PC Refinement, Pattern Match, Phase Map, EDS, Analysis,
  Crystal Database, Simulation, and more).
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the Electron
  shell, React frontend, FastAPI backend, and scientific layer fit together.
- **Per-directory READMEs:** [backend/](backend/README.md),
  [frontend/](frontend/README.md), [electron/](electron/README.md),
  [analysis/](analysis/README.md), [simulation/](simulation/README.md),
  [tools/](tools/README.md), and [docs/](docs/README.md) describe each part of
  the codebase.

## Sample data

**Crystal structures ship with the repository.** [`sample_data/phases/`](sample_data/)
contains four open-licensed (CC0, Crystallography Open Database) structures — Al,
Al₂CuMg, Al₇Cu₂Fe and α-(Al,Mn,Si) — each as `.cif`, `.xtal` and `.sht`, so you can
try Hough and Spherical indexing without simulating anything first. Dictionary
indexing needs a master pattern, which you simulate from the `.xtal` (see
[Simulation](docs/user-guide/Simulation.md)). See
[`sample_data/README.md`](sample_data/README.md) for provenance and how to load them.

**EBSD measurements do not** — they are ~800 MB and are distributed through Zenodo
instead ([10.5281/zenodo.21021937](https://doi.org/10.5281/zenodo.21021937); see
[`sample_data/README.md`](sample_data/README.md)). Or simply point the EBSD Viewer
at your own H5OINA/H5 files.

## Getting help

- **Questions** (how to do something, or whether a result looks right):
  [Discussions → Q&A](https://github.com/SeSam-MUL/Orienta/discussions/categories/q-a).
  Answers are given voluntarily and do not replace your own validation of your results.
- **Something fails or is clearly wrong:** open a
  [bug report](https://github.com/SeSam-MUL/Orienta/issues/new?template=bug_report.yml).
  The **⚠ Report a problem** button in the app collects the version and the logs for you;
  see [Reporting a problem](docs/user-guide/ReportingProblems.md).
- **Ideas:** a [feature request](https://github.com/SeSam-MUL/Orienta/issues/new?template=feature_request.yml),
  or [Discussions → Ideas](https://github.com/SeSam-MUL/Orienta/discussions/categories/ideas).
- **Security problems:** please report them privately, as described in [SECURITY.md](SECURITY.md).

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Known limitations & to be done

Orienta is beta software. This section lists what we know is missing, incomplete,
or misleading — so you find it here rather than in your results. Items marked
**wrong result** can hand you a plausible-looking answer that is not correct.

### Data and results

- **EDAX hexagonal-grid scans are resampled to a square grid.** Orienta detects an
  EDAX H5 file whose header says `Grid Type = HexGrid` and loads it as a square map:
  each square pixel takes the pattern of the nearest measured point (nearest
  neighbour; patterns are never interpolated). The square step equals the in-row
  spacing of the hex grid and the map covers the extent of the scan. The result is a
  resampled view of the scan, not the measured points themselves: some measured
  points are not used, and indexing, step size, pixel counts and grain statistics
  all refer to the resampled grid. The loader reads the row layout from the stored
  point positions and refuses the file with an error if that layout is not the
  alternating long/short rows of an EDAX hex scan, if the X/Y positions are missing,
  or if the number of measured points does not match the number of stored patterns.
  Resampling applies to EDAX H5 files only. An EDAX UP1/UP2 file is refused with an
  error when its version-3 header or its `.osc` file shows a hexagonal grid. A
  version-1 file without its `.osc` cannot be checked, so keep the `.osc` next to it,
  or export hexagonal scans as H5 from OIM. This detection has been tested with
  synthetic headers only. Square-grid EDAX H5 files, and square-grid UP1/UP2 files
  with an `.osc` sidecar, load as usual, without resampling.
- **The "Texture Components" map layer is empty.** The layer can be selected on
  the Analysis page but always returns a blank map. The per-component numbers in
  the texture table and the Excel export are real; only the spatial map is a stub.
- **"Texture index" and "entropy" are not computed from a true ODF.** They are
  derived from texture-component volume-fraction bins, which approximate an ODF.
  Treat them as a comparative measure between your own datasets, not as values
  comparable with MTEX or other ODF implementations. There are no pole figures on
  the Analysis page (the pole-figure tool lives on the Phase Map page).
- **Two recrystallization statistics are placeholders.** `low_gkam_fraction` and
  `low_kam_fraction` are reported as `0.0` and should be ignored. The RX fraction,
  grain counts and the high-BC fraction are computed for real.
- **Automated indexing and phase identification must be validated.** This is not a
  defect but a property of the method — see the Disclaimer below.

### Simulation

- **The EMsoft/EMSphInx simulation engine is not usable in this release.** The
  in-app installer builds EMsoft and EMSphInx correctly, but the automation module
  and configuration template that drive them are not part of this repository, so
  starting a simulation fails with a configuration error. The **EMsoft-free GPU
  forward model** is unaffected and works.
- **The installer wizard has rough edges.** On WSL 1 it asks you to run one command
  by hand. Some failures (a declined Windows elevation prompt, for instance) are
  reported as if they had succeeded, so watch the build log rather than the status
  line. If GPU simulation does not work after a successful install, check the
  OpenCL vendor files under `/etc/OpenCL/vendors/` — the wizard can leave them with
  the wrong contents.

### Installation

These concern installing from source; the ready-made packages under
[Download](#download) are not affected.

- **Requires Node.js 20.19+**, not 18, despite what older instructions said. The
  frontend build fails on Node 18.
- **`requirements.txt` installs the CUDA runtime even on machines without an
  NVIDIA GPU** (several GB). Everything still runs on the CPU; it is wasted
  download, not a failure.
- **Python 3.11 is the only tested version.** The dependency that used to stop the
  installation on Python 3.12 has been removed, but 3.12 and 3.13 have not been
  tested yet. Use Python 3.11.
- **A ZIP or Zenodo download reports its version as "unknown"** under
  *Settings → About Orienta*, because the version is read from git metadata that a
  ZIP does not carry. Clone the repository if you need the version to be visible —
  for example when reporting a problem.
- **53 of the backend tests skip** without the measurement files and crystal
  database they need; neither is distributed. The remaining 754 run offline.

### To be done

In rough priority order: ship the simulation automation module, or remove the
engine from the UI · fix the installer's OpenCL step and make failures surface as
failures · wire the texture component map to its existing implementation and
compute the two placeholder RX statistics · move the CUDA packages behind an
optional requirements file and test Python 3.12 and 3.13 · write the version
into the release archive so downloads identify themselves.

If you hit something that is not on this list, please report it — the **⚠ Report a problem**
button in the app collects the logs for you; see
[docs/user-guide/ReportingProblems.md](docs/user-guide/ReportingProblems.md).

## Running from source

Only needed if you want to change the code, or run Orienta on a system the
packages above do not cover.

### Requirements

- **Python 3.11** in a conda environment named `ebsd`.
- **Node.js 20.19+** (or 22.12+) — the build toolchain (Vite 8, Vitest 4)
  requires it. Only needed to build the frontend / run it in dev.
- Optional: **WSL** with **EMsoft** / **EMSphInx** for master-pattern simulation
  and spherical indexing. You do **not** install these by hand — Orienta ships a
  wizard under **Settings → EMsoft + EMSphInx Installation** that sets up WSL,
  the Linux user and both tools. See
  [INSTALL.md § Step 5](INSTALL.md#step-5--optional-emsoft--emsphinx-via-the-built-in-installer).
  An NVIDIA GPU (CUDA 12) is optional — everything runs on the CPU without one,
  only slower. Note that `requirements.txt` currently installs the CUDA runtime
  regardless (several GB); see [Known limitations](#known-limitations--to-be-done).

### Install from source

> **New here?** Follow the full beginner walkthrough in **[INSTALL.md](INSTALL.md)** —
> it covers prerequisites (with download links), the Python environment, building
> the UI, both run modes, and troubleshooting, step by step.

Quick version, with conda and Node.js 20.19+ installed.

> **The first command is `cd`, and it is not optional.** A fresh Miniforge or Anaconda Prompt
> starts in your home folder, *not* in Orienta. Every command below — especially
> `pip install -r requirements.txt` — only works from the folder that contains
> `requirements.txt`. Replace the example path with wherever you unzipped or
> cloned Orienta.

```bash
# 1. Go into the Orienta folder FIRST (adjust the path to your machine):
cd C:\Users\YourName\Downloads\Orienta      # Windows
# cd ~/Downloads/Orienta                    # macOS / Linux

# 2. Confirm you are in the right place — this must list requirements.txt:
dir requirements.txt                        # Windows
# ls requirements.txt                       # macOS / Linux

# 3. Python environment (Miniforge recommended; on Anaconda/Miniconda the
#    --override-channels keeps Python out of Anaconda's licensed repository,
#    see INSTALL.md Step 2)
conda create -n ebsd python=3.11 --override-channels -c conda-forge
conda activate ebsd
# Optimized BLAS first (NumPy is ~1000x slower on reference BLAS):
conda install --override-channels -c conda-forge "blas=*=openblas"
pip install -r requirements.txt
# macOS: replace the four lines above with
#   conda env create -f environment-macos.yml && conda activate ebsd
# (pip on a Mac ends in "OMP: Error #15", see INSTALL.md)

# 4. Build the user interface (needed before first run):
cd frontend && npm install && npm run build && cd ..
```

If step 2 says the file cannot be found, you are in the wrong folder — `cd` to
the correct one before continuing.

### Run

Both run modes use the conda `ebsd` environment (the Electron desktop shell
launches the same Python backend behind the scenes).

```bash
python start_app.py        # Browser version — opens http://127.0.0.1:8000
python start_desktop.py    # Desktop version — native Electron window
```

| Mode | Command | Notes |
|------|---------|-------|
| **Browser** | `python start_app.py` | Starts the backend and opens the app at http://127.0.0.1:8000 |
| **Desktop** | `python start_desktop.py` | Native Electron window (needs Node + the frontend build) |
| Browser, headless | `python start_app.py --headless` | Backend keeps running if you close the browser — for long batches |
| Dev backend | `python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000` | |
| Dev frontend | `cd frontend && npm run dev` (http://localhost:5173) | or `python start_app.py --dev` |
| Electron (dev) | `cd frontend && npm run electron:dev` | |

> The backend has no auto-reload — restart it after Python changes.

See **[INSTALL.md](INSTALL.md)** for prerequisites and troubleshooting.

### Tests

```bash
pytest tests/ -v          # Python backend
cd frontend && npm test   # Vitest (frontend)
```

### Architecture

```
Electron shell        electron/main.js, electron/preload.js
      │
React frontend        frontend/  (Vite build, ~20 feature pages)
      │  HTTP + WebSocket
FastAPI backend       backend/api/  (route files under backend/api/routes/)
      │
Scientific layer      kikuchipy, orix, diffsims, PyEBSDIndex, numpy/scipy, PyTorch (GPU)
```

## Disclaimer

Orienta is **research software**, provided **free of charge** and **without any
warranty**, under the GNU GPL-3.0-or-later (see [LICENSE](LICENSE), §15
*Disclaimer of Warranty* and §16 *Limitation of Liability*). In particular:

- The software is provided **"as is"**; the entire risk as to its quality and
  performance is with you.
- To the maximum extent permitted by applicable law, the authors, copyright
  holders, and their institutions accept **no liability for any damages**
  arising from the use of (or inability to use) this software — including,
  without limitation, **loss of or damage to data**, inaccurate results, or any
  direct, indirect, incidental, or consequential losses.
- All processing happens **locally on your own machine**. Your data is never
  transmitted to the authors; there is no hosted service behind this
  application.
- Orienta is designed to treat your measurement files as **read-only** —
  analysis results, exports, and simulation outputs are written to separate
  files. Nevertheless, no software is free of defects: **always keep backups of
  your original data**.
- Results produced by automated indexing, phase identification, and
  quantification **must be independently validated** before you rely on them in
  publications, engineering, or any safety-relevant decisions.

## License

**GPL-3.0-or-later** — see [LICENSE](LICENSE). Third-party and derived-code
attributions are in [NOTICE.md](NOTICE.md). This project derives from GPL-3.0
code (kikuchipy), so the combined work is GPL-3.0. Use of the software is
subject to the warranty disclaimer and liability limitation above.

## How to cite

If you use this software in your research, please cite it via its **concept DOI**,
which always resolves to the most recent archived version:

> Samberger, S., Pogatscher, S., Kobayashi, E., & Weißensteiner, I. (2026). Orienta. Zenodo. https://doi.org/10.5281/zenodo.22664080

If you need to record the exact version you worked with, cite that release's own
DOI instead — every Zenodo version has one, listed on the record page.

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff); GitHub's
"Cite this repository" button reads it automatically.

**Please also cite the underlying methods** when you use the corresponding features
(see [NOTICE.md](NOTICE.md) → "Algorithm / method attributions"):

- **Spherical (SHT) indexing** — Lenthe, Singh &amp; De Graef, *Ultramicroscopy* **207**,
  112841 (2019); and the [EMSphInx](https://github.com/EMsoft-org/EMSphInx) project.
  Orienta's GPU spherical indexer is an independent reimplementation of this method.
- **Hough indexing** — the Radon-transform band detection and the band indexing
  are done by [PyEBSDIndex](https://github.com/USNavalResearchLaboratory/PyEBSDIndex)
  (U.S. Naval Research Laboratory). Please cite Rowenhorst, D. J., Callahan, P. G.
  &amp; Ånes, H. W., "Fast Radon transforms for high-precision EBSD orientation
  determination using PyEBSDIndex", *J. Appl. Cryst.* **57**(1), 3–19 (2024),
  [doi:10.1107/S1600576723010221](https://doi.org/10.1107/S1600576723010221).
  Wherever kikuchipy is cited for a Hough-indexed result, cite PyEBSDIndex with it,
  and also when the pattern centre was refined with Orienta's PC refinement, which
  uses PyEBSDIndex's indexing and PC optimiser.
- **Dictionary indexing, simulation &amp; projection** — [kikuchipy](https://kikuchipy.org/)
  and [EMsoft](https://github.com/EMsoft-org/EMsoft).

## Funding &amp; acknowledgments

This work was supported by:

- The **European Research Council (ERC)** under the European Union's Horizon Europe
  research and innovation programme — the **HETEROCIRCAL** project (grant agreement No. **101124514**).
- The **Christian Doppler Research Association (CDG)** — **Christian Doppler
  Laboratory for Deformation-Precipitation Interactions in Aluminum Alloys** and its industrial partner(s).
- The **Canon Foundation in Europe**.

> Funded by the European Union. Views and opinions expressed are however those of
> the author(s) only and do not necessarily reflect those of the European Union or
> the European Research Council Executive Agency. Neither the European Union nor the
> granting authority can be held responsible for them.

