# Third-party license texts

Reproduced here because the licenses require the notice to travel with every
copy, and the files that contain the derived code or data cannot always carry
the full text. `NOTICE.md` at the repository root says which file each text
belongs to.

| File | Covers |
|------|--------|
| `EMsoft-License.txt` | tables and routines transcribed from EMsoft (`backend/forward_sim/`) |
| `SHTfile-License.txt` | tables and packing routines ported from SHTfile (`backend/forward_sim/io/sht_writer.py`, `backend/spherical_gpu/pipeline/sht_io.py`) |
| `torch-dct-License.txt` | the weighted DCT/DST layers in `backend/spherical_gpu/_math/sht.py` |
| `Feather-License.txt` | the SVG icons in `crystal-structures-for-ebsd-main/calculationxtal/icons/` |
| `python-build-standalone/` | components of the standalone CPython the Windows and Linux installers download |
| `micromamba-License.txt` | the micromamba binary the macOS installer downloads and keeps in the runtime folder |

This directory is part of every release package (`scripts/build_runtime_package.py`),
and the release build refuses a package whose `NOTICE.md` points at a file that
is not in it.
