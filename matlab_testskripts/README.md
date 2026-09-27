# MATLAB/MTEX reference scripts

**Author:** Dr. Irmgard Weißensteiner, Montanuniversität Leoben.
**License:** GNU General Public License, version 2 or (at your option) any later
version — see [LICENSE](LICENSE) in this folder. `freezeColors.m` is the one
exception; see below.

## What these scripts are

These are the MATLAB scripts, built on the [MTEX](https://mtex-toolbox.github.io/)
toolbox, that the Python module [`analysis/`](../analysis/) was ported from. They
describe the EBSD post-processing workflow Orienta reproduces: import and
clean-up, grain reconstruction, grain size, deformation (KAM, grain boundaries,
band contrast), texture (ODF, texture components, surface planes),
recrystallization, and the `Documentation.xlsx` batch output.

They are kept as the **reference** for that port, so that anyone can check what
Orienta computes against the original. They are not part of the running
application and Orienta never calls them; use the Python module for actual work.

**This copy is a snapshot.** The author has developed her scripts further since
these were handed over; this folder holds the state that Orienta rebuilt, not
her current version. Do not treat it as the latest word on the workflow.

## Files

| File | Purpose | Python counterpart |
|------|---------|--------------------|
| `EBSDanalysis_frame.m` | Batch driver: loops over the data files of a folder, band-contrast histogram with the 3-Gaussian fit, writes `Documentation.xlsx` | `analysis/batch_processor.py`, `analysis/bc_analysis.py` |
| `ImportAndModify_MTEX6.m` | Load, crop, tilt correction, particle removal, denoising, grain reconstruction | `analysis/ebsd_dataset.py`, `analysis/grain_analysis.py`, `analysis/smoothing.py` |
| `GrainSizeAnalysis_mtex6.m` | Equivalent circle diameter, maximum dimensions, aspect ratio, histograms | `analysis/grain_analysis.py` |
| `DeformationAnalysis.m` | KAM, grain-boundary classification and lengths, band-contrast maps | `analysis/deformation_analysis.py` |
| `TextureAnalysis_mtex6.m` | ODF, texture components, surface-plane fractions | `analysis/texture_analysis.py`, `analysis/texture_components.py` |
| `RxxAnalysis.m` | Recrystallization criteria (GOS, grain-average BC and KAM) | `analysis/rx_analysis.py` |
| `gKAM.m`, `gBC.m` | Grain-average KAM and band contrast | `analysis/rx_analysis.py` (`calculate_grain_average_kam`), `analysis/bc_analysis.py` (`grain_average_bc`) |
| `excelColumn.m` | Column index to Excel column letters | `analysis/excel_exporter.py` |
| `WhiteBlueHeatColorMap.m`, `b2wcontrast.m` | Colour maps for the figures | — |
| `freezeColors.m` | Third-party utility (John Iversen), see below | — |

`EBSDanalysis_frame.m` contains two placeholders (`<path to your MTEX
installation>`, `<path to this scripts folder>`) where the original had the
author's local paths.

## Running them

You need MATLAB and an installation of MTEX (the scripts were last used with
MTEX 6). MTEX is **not** included here; install it from
<https://mtex-toolbox.github.io/> and put its path into `EBSDanalysis_frame.m`.
Without MTEX the scripts cannot run.

## Licensing details

- Every script by the author carries a four-line header naming her and the
  license (`SPDX-License-Identifier: GPL-2.0-or-later`). The header is the only
  change made to the files for publication; the code below it is as received,
  byte for byte. The header is pure ASCII (the name is spelled
  "Weissensteiner, Montanuniversitaet" there) so that every file keeps exactly
  its original encoding and line endings: two are Windows-1252, ten of twelve
  use Windows line endings, and none changed class. A MATLAB installation that
  read them before reads them unchanged. Because the header sits above the
  `function` line, `help` on the small helpers (`gKAM`, `gBC`, `excelColumn`,
  the two colour maps) prints it before their own help text.
- The Python module cites these scripts by line number ("Reference:
  DeformationAnalysis.m line 147" and the like). Those numbers count the
  header, so they match the files as published here.
- The copyright year in `LICENSE` and the headers is the year the scripts
  entered Orienta's history (first commit 2026-02-25). The scripts themselves
  are older; the author may replace it with the actual years.
- `freezeColors.m` (v2.3) is by John Iversen and is **not** covered by the
  author's license. Its own header carries its terms ("Free for all uses, but
  please retain the following: Original Author: John Iversen, 2005-10"); the
  upstream project later adopted the MIT license, whose text is kept in
  [`../licenses/freezeColors-Upstream-MIT.txt`](../licenses/freezeColors-Upstream-MIT.txt).
- MTEX is GPL-2 software distributed separately by its authors and is not
  redistributed here. See the MATLAB section of [`../NOTICE.md`](../NOTICE.md)
  for the MTEX citations.
