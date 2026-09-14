# Sample data

Example data so you can try Orienta end-to-end without your own measurements.

## Crystal structures — `phases/`
Four open-licensed (CC0, from the Crystallography Open Database) structures
covering the main phases in the sample alloys:

- `Al.cif` — aluminium matrix (fcc)
- `Al7Cu2Fe.cif` — Al₇Cu₂Fe intermetallic
- `Al2CuMg.cif` — Al₂CuMg (S-phase)
- `alpha-AlMnSi.cif` — cubic α-(Al,Mn,Si) phase

See [`phases/PROVENANCE.md`](phases/PROVENANCE.md) for sources, licenses and
indexing coverage. **To use them:** open *Crystal Database → Load Files* (or copy
them into your `Database/CIF_Library/`), then generate the `.xtal` / master / SHT
as needed for Hough / Dictionary / Spherical indexing.

## EBSD datasets — `patterns/` (Zenodo archive only)
The example EBSD measurements are large, so they are included in the **Zenodo
archive**, not in the source-code repository:

| File | Sample | Notes |
|------|--------|-------|
| `SampleB_Al-extrusion.h5oina` | Al extrusion alloy | Oxford H5OINA, includes EDS |
| `Scan1_indexed.h5` | Scan 1 | pre-indexed dataset |
| `AA7050_R_area1.h5oina` | AA7050 (Al-Zn-Mg-Cu) | 39×136, small & fast to index |

These are the authors' own measurements, shared for testing. Load any of them via
the **EBSD Viewer → Load Data**.
