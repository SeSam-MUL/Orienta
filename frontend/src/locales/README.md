# UI translations (i18n)

One folder per language: `en` (source of truth), `de`, `ja`, `zh`.
One JSON file per namespace (UI area): `common`, `nav`, `shell`, and one per
page (`eds`, `indexing`, …). Files are auto-loaded by `src/i18n/index.js`.

## Conventions
- Keep technical acronyms untranslated everywhere: **EBSD, EDS, PC, IPF, KAM,
  GOS, HDF5, H5OINA, CIF, XTAL, DWF, NCC, ODF, MTEX, EMsoft, EMSphinx, Hough,
  Dictionary, Spherical, FAISS, GPU, ML**.
- Use `{{var}}` interpolation, never string concatenation.
- `common` holds app-wide verbs/nouns (Save, Cancel, Export …). Reuse them.

## Status
- `en` / `de`: authored.
- `ja` / `zh`: machine-quality first pass — **needs native/technical review**
  before public release.
