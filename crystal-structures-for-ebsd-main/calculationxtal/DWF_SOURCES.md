# Debye–Waller Factors (DWF.xlsx) — sources & provenance

`DWF.xlsx` maps an element symbol to its **Debye–Waller factor B at 300 K**,
used during CIF → `.xtal` conversion (`convert-cif-to-xtal`). The value is the
isotropic B-factor in **nm²** (B = 8π²⟨u²⟩); B[nm²] = B[Å²] × 0.01.
Example: Al = 0.007993 nm² = 0.7993 Å².

If an element is **not** in the table, the converter falls back to a generic
`0.005` nm² — see `Xtal_Generator_GUI.py` / `database.py`
(`dwf_map.get(symbol, 0.005)`). Add missing elements with a real, cited value.

## Source of the current values

All values are **B at 300 K from Peng et al. (1996), Table 1**, transcribed
from the deposited supplement (IUCr SUP82472, article zh0008):

> **Peng, L.-M., Ren, G., Dudarev, S. L. & Whelan, M. J. (1996).**
> *Debye–Waller factors and absorptive scattering factors of elemental
> crystals.* Acta Cryst. **A52**, 456–470. doi:10.1107/S0108767396005382

Peng Table 1 covers **44 elemental crystals** over 1–1000 K (from phonon DOS,
2–3 % accuracy). 38 of them have a solid value at 300 K and are loaded here;
the four noble gases (Ne, Ar, Kr, Xe) have no 300 K solid value and are omitted.

The six pre-existing legacy values (Al, Si, Mg, Ni, Fe, Mn) were checked
against Peng during this work:
- **Al, Si, Mg, Ni** — exact match to Peng @ 300 K (confirmed provenance).
- **Fe** — the legacy value `0.5771` matched **neither** Peng phase
  (Fe-BCC 0.3328, Fe-FCC 0.5710) and was an error. Corrected to the
  **BCC** value **0.3328** (α-ferrite is the room-temperature stable phase;
  the FCC value at 300 K is an extrapolation of a non-stable phase). For Fe in
  intermetallics the true B is site-specific, but BCC is the RT-grounded
  reference and the DWF only weakly affects the simulated pattern.
- **Mn** — **not present in Peng 1996** (it has no Mn or Co). Kept at its
  legacy value `0.0048` and flagged `unverified` in the Reference column. If a
  cited Mn (or Co) value is needed, see Butt, Bashir, Willis & Heger (1988),
  Acta Cryst. **A44**, 396 (cubic-element compilation), or another source.

### Alternative phase values (not used, for reference)
- Fe (FCC) @ 300 K = 0.5710 Å² = 0.005710 nm²
- Ca (BCC) @ 300 K = 2.6615 Å²; Ca (FCC, used) = 2.0443 Å²

## How to add / edit entries

Use the **Debye-Waller Factors** dialog in the Crystal Database page (inline-
editable value + Reference, plus a “+ Add” row). It upserts via
`PATCH /api/database/dwf`. Always fill the `Reference` column with the citation
so every value keeps its provenance.
